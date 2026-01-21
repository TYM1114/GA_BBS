#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <limits>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <unordered_map>

// Load Modules
#include "DataLoader.h"
#include "YardSystem.h"

// --- Parameter Settings ---
const int POPULATION_SIZE = 50;
const int MAX_GENERATIONS = 100;
const double MUTATION_RATE = 0.2;
const int BEAM_WIDTH = 5; // change to smaller value if runtime is too long

// --- Output Format Definition ---
struct MissionLog {
    int mission_no;
    std::string mission_type;     // "target", "block", or "return"
    int batch_id;
    int container_id;
    Coordinate src;
    Coordinate dst;               // If type is target, dst is (-1,-1,-1) / Workstation
    int mission_priority;
    std::string mission_status;   // "PLANNED"
    long long created_time;
};

// ACO Module: Pheromone Manager (4D Matrix)

class PheromoneManager {
private:
    // Dimensions: [Row][Bay][Level][Step_Index]
    std::vector<std::vector<std::vector<std::vector<double>>>> matrix;
    int maxRows, maxBays, maxTiers, maxSteps;
    double evaporationRate = 0.1;

public:
    PheromoneManager(int r, int b, int t, int s) 
        : maxRows(r), maxBays(b), maxTiers(t), maxSteps(s) {
        // Initialize all pheromone values to 1.0
        matrix.assign(r, std::vector<std::vector<std::vector<double>>>(
            b, std::vector<std::vector<double>>(
                t, std::vector<double>(s, 1.0))));
    }

    double getPheromone(int r, int b, int t, int step) const {
        if (r < 0 || r >= maxRows || b < 0 || b >= maxBays || t < 0 || t >= maxTiers) return 1.0;
        int sIdx = step % maxSteps; // Use modulo to handle long sequences
        return matrix[r][b][t][sIdx];
    }

    void update(const std::vector<MissionLog>& bestLogs, int totalCost) {
        // 1. Evaporation: Decrease pheromone levels over time
        for (auto& row : matrix)
            for (auto& bay : row)
                for (auto& tier : bay)
                    for (double& val : tier) val *= (1.0 - evaporationRate);

        // 2. Reinforcement: Increase pheromone for successful paths
        if (totalCost <= 0) return;
        double deposit = 100.0 / (totalCost + 1);
        
        for (const auto& log : bestLogs) {
            if (log.dst.row != -1) { // Record successful destination coordinates
                int sIdx = log.mission_no % maxSteps;
                if (log.dst.row < maxRows && log.dst.bay < maxBays && log.dst.tier < maxTiers) {
                    matrix[log.dst.row][log.dst.bay][log.dst.tier][sIdx] += deposit;
                }
            }
        }
    }
};

// Core Module 1: BBS Evaluator (Revised: With Lookahead Penalty)

class BBS_Evaluator {
public:
    struct SearchNode {
        YardSystem yard;
        int g; // Actual Cost
        int f; // Total Score (g + penalty)
        bool operator<(const SearchNode& other) const { return f < other.f; }
    };

    struct LogNode {
        YardSystem yard;
        int g;
        int f;
        std::vector<MissionLog> history;
        bool operator<(const LogNode& other) const { return f < other.f; }
    };

    // Helper: Calculate Move Penalty with ACO Guidance
    static int calculateMovePenalty(const YardSystem& yard, int r, int b, 
                                    const std::unordered_map<int, int>& priorityMap, 
                                    int currentSeqIndex,
                                    const PheromoneManager* pm = nullptr) {
        
        int currentHeight = yard.tops[r][b];
        int topTier = currentHeight - 1;
        int penalty = 0;

        // 1. Lookahead Penalty: Protect future targets
        int minBelowPriority = std::numeric_limits<int>::max();
        bool foundFutureTarget = false;
        if (topTier >= 0) {
            for (int t = 0; t <= topTier; ++t) {
                int boxId = yard.grid[r][b][t];
                auto it = priorityMap.find(boxId);
                if (it != priorityMap.end()) {
                    int p = it->second;
                    if (p >= currentSeqIndex) {
                        if (p < minBelowPriority) { 
                            minBelowPriority = p; 
                            foundFutureTarget = true; 
                        }
                    }
                }
            }
        }
        if (foundFutureTarget) {
            int distance = minBelowPriority - currentSeqIndex;
            penalty += 1000 + (100000 / (distance + 1)); 
        }

        // 2. Height Penalty: Guide towards flatter layout
        if (currentHeight >= yard.MAX_TIERS - 1) penalty += 5000;
        else penalty += (currentHeight * 25);

        // 3. ACO Guidance: Divide penalty by pheromone value
        if (pm != nullptr) {
            double ph = pm->getPheromone(r, b, currentHeight, currentSeqIndex);
            if (ph > 0) penalty = static_cast<int>(penalty / ph);
        }
        return penalty;
    }

    // -------------------------------------------------------------------------
    // Helper: Find Best Return Slot
    // -------------------------------------------------------------------------
    static Coordinate findBestReturnSlot(const YardSystem& yard, int targetId, 
                                         const std::unordered_map<int, int>& priorityMap, 
                                         int currentSeqIndex,
                                         const PheromoneManager* pm = nullptr) {
        Coordinate bestPos = {-1, -1, -1};
        int minPenalty = std::numeric_limits<int>::max();

        for (int r = 0; r < yard.MAX_ROWS; ++r) {
            for (int b = 0; b < yard.MAX_BAYS; ++b) {
                if (!yard.canReceiveBox(r, b)) continue;

                int penalty = calculateMovePenalty(yard, r, b, priorityMap, currentSeqIndex, pm);

                int topTier = yard.tops[r][b] - 1;
                if (topTier >= 0) {
                    int boxBelowId = yard.grid[r][b][topTier];
                    if (boxBelowId < targetId) penalty += 50; 
                    else penalty += yard.tops[r][b]; 
                } else {
                    penalty += 20; 
                }

                if (penalty < minPenalty) {
                    minPenalty = penalty;
                    bestPos = {r, b, yard.tops[r][b]};
                }
            }
        }
        return bestPos;
    }

    static int evaluate(const YardSystem& initialYard, const std::vector<int>& retrievalSequence, const PheromoneManager* pm = nullptr) {
        return run_internal_logic(initialYard, retrievalSequence, pm);
    }
    // -------------------------------------------------------------------------
    // 2. Execute and Record (For CSV Output)
    // -------------------------------------------------------------------------
    static std::vector<MissionLog> solveAndRecord(const YardSystem& initialYard, const std::vector<int>& retrievalSequence,const PheromoneManager* pm = nullptr) {
        std::vector<LogNode> currentBeam;
        currentBeam.push_back({initialYard, 0, 0, {}}); // g=0, f=0

        int missionSerial = 1;
        long long baseTime = 1705363200; 

        // Create Priority Map (ID -> Sequence Index)
        std::unordered_map<int, int> priorityMap;
        for(size_t i=0; i<retrievalSequence.size(); ++i) {
            priorityMap[retrievalSequence[i]] = (int)i;
        }

        // Iterate through each target box
        for (int i = 0; i < retrievalSequence.size(); ++i) {
            int targetId = retrievalSequence[i];
            
            
            // Phase 1: Outbound (Move Target to Workstation)
            
            
            std::vector<LogNode> finishedBeam;
            std::vector<LogNode> processingBeam = currentBeam;

            int depthSafety = 0;
            while (!processingBeam.empty()) {
                std::vector<LogNode> nextStepBeam;

                for (const auto& node : processingBeam) {
                    // Case A: Target is at the top -> Retrieve
                    if (node.yard.isTop(targetId)) {
                        LogNode doneNode = node;
                        Coordinate srcPos = doneNode.yard.getBoxPosition(targetId);
                        doneNode.yard.removeBox(targetId);
                        
                        MissionLog m;
                        m.mission_no = missionSerial++;
                        m.mission_type = "target";
                        m.batch_id = 20260117;
                        m.container_id = targetId;
                        m.src = srcPos;
                        m.dst = {-1, -1, -1}; // Workstation
                        m.mission_priority = 0;
                        m.mission_status = "PLANNED";
                        m.created_time = baseTime;
                        
                        doneNode.history.push_back(m);
                        
                        // Reset f value, as Phase 1 ends and we don't need previous penalties for Phase 2
                        doneNode.f = doneNode.g; 
                        
                        finishedBeam.push_back(doneNode);
                    } 
                    // Case B: Target is blocked -> Move blockers
                    else {
                        std::vector<int> blockers = node.yard.getBlockingBoxes(targetId);
                        if (blockers.empty()) continue; 

                        int blockerId = blockers.back();
                        Coordinate srcPos = node.yard.getBoxPosition(blockerId);

                        for (int r = 0; r < node.yard.MAX_ROWS; ++r) {
                            for (int b = 0; b < node.yard.MAX_BAYS; ++b) {
                                if (r == srcPos.row && b == srcPos.bay) continue;

                                YardSystem newYard = node.yard;
                                if (newYard.moveBox(srcPos.row, srcPos.bay, r, b)) {
                                    LogNode newNode = node;
                                    newNode.yard = newYard;
                                    newNode.g += 1; // Increase actual cost

                                    // [CRITICAL] Calculate Penalty: Does this move block a future target?
                                    int penalty = calculateMovePenalty(node.yard, r, b, priorityMap, i);
                                    
                                    // Sorting Score = Actual Cost + Penalty
                                    newNode.f = newNode.g + penalty;

                                    MissionLog m;
                                    m.mission_no = missionSerial++;
                                    m.mission_type = "block";
                                    m.batch_id = 20260117;
                                    m.container_id = blockerId;
                                    m.src = srcPos;
                                    m.dst = {r, b, newNode.yard.getBoxPosition(blockerId).tier};
                                    m.mission_priority = 0;
                                    m.mission_status = "PLANNED";
                                    m.created_time = baseTime;

                                    newNode.history.push_back(m);
                                    nextStepBeam.push_back(newNode);
                                }
                            }
                        }
                    }
                }
                
                // Pruning (Phase 1)
                if (!nextStepBeam.empty()) {
                    std::sort(nextStepBeam.begin(), nextStepBeam.end());
                    if (nextStepBeam.size() > BEAM_WIDTH) nextStepBeam.resize(BEAM_WIDTH);
                }
                processingBeam = nextStepBeam;
                if (++depthSafety > 30) break; 
            }

            if (finishedBeam.empty()) return {}; // Dead End

            // Use g (actual cost) or f to select best results for Phase 2
            std::sort(finishedBeam.begin(), finishedBeam.end());
            if (finishedBeam.size() > BEAM_WIDTH) finishedBeam.resize(BEAM_WIDTH);

            
            // Phase 2: Inbound (Return Target to Yard)
            
            
            std::vector<LogNode> returnPhaseBeam;

            for (const auto& node : finishedBeam) {
                // Find best return slot (Using Priority Map to avoid blocking future targets)
                Coordinate bestSlot = findBestReturnSlot(node.yard, targetId, priorityMap, i);

                if (bestSlot.row != -1) {
                    LogNode returnNode = node;
                    returnNode.yard.initBox(targetId, bestSlot.row, bestSlot.bay, bestSlot.tier);
                    
                    MissionLog m;
                    m.mission_no = missionSerial++;
                    m.mission_type = "return";
                    m.batch_id = 20260117;
                    m.container_id = targetId;
                    m.src = {-1, -1, -1};
                    m.dst = bestSlot;
                    m.mission_priority = 0;
                    m.mission_status = "PLANNED";
                    m.created_time = baseTime;

                    returnNode.history.push_back(m);
                    
                    // Return action does not increase g (usually), but reset f
                    returnNode.f = returnNode.g; 

                    returnPhaseBeam.push_back(returnNode);
                }
            }

            if (returnPhaseBeam.empty()) return {}; 
            currentBeam = returnPhaseBeam;
        }

        if (currentBeam.empty()) return {};
        
        auto finalLogs = currentBeam[0].history;
        for(size_t i=0; i<finalLogs.size(); ++i) {
            finalLogs[i].mission_no = (int)(i + 1);
            finalLogs[i].mission_priority = (int)(i + 1);
            finalLogs[i].created_time += (i * 30); 
        }
        return finalLogs;
    }

private:
    static int run_internal_logic(const YardSystem& initialYard, const std::vector<int>& retrievalSequence, const PheromoneManager* pm) {
         std::vector<SearchNode> currentBeam;
         currentBeam.push_back({initialYard, 0, 0});
         
         std::unordered_map<int, int> priorityMap;
         for(size_t i=0; i<retrievalSequence.size(); ++i) priorityMap[retrievalSequence[i]] = (int)i;

         for (int i = 0; i < (int)retrievalSequence.size(); ++i) {
            int targetId = retrievalSequence[i];
            std::vector<SearchNode> finishedBeam;
            std::vector<SearchNode> processingBeam = currentBeam;
            int depth = 0;
            
            while(!processingBeam.empty()) {
                std::vector<SearchNode> nextStep;
                for(const auto& node : processingBeam) {
                    // Aggressive Pruning: Discard high-cost paths
                    if (node.g > 400) continue; 

                    if(node.yard.isTop(targetId)) {
                        SearchNode dn = node; 
                        dn.yard.removeBox(targetId);
                        dn.f = dn.g; 
                        finishedBeam.push_back(dn);
                    } else {
                        auto blks = node.yard.getBlockingBoxes(targetId);
                        if(blks.empty()) continue;
                        int bid = blks.back();
                        Coordinate pos = node.yard.getBoxPosition(bid);
                        for(int r=0; r<node.yard.MAX_ROWS; ++r) {
                            for(int b=0; b<node.yard.MAX_BAYS; ++b) {
                                if(r==pos.row && b==pos.bay) continue;
                                YardSystem ny = node.yard;
                                if(ny.moveBox(pos.row, pos.bay, r, b)) {
                                    int penalty = calculateMovePenalty(ny, r, b, priorityMap, i, pm);
                                    nextStep.push_back({ny, node.g+1, node.g+1+penalty});
                                }
                            }
                        }
                    }
                }
                if(!nextStep.empty()) {
                    std::sort(nextStep.begin(), nextStep.end());
                    if(nextStep.size() > (size_t)BEAM_WIDTH) nextStep.resize(BEAM_WIDTH);
                }
                processingBeam = nextStep;
                if(++depth > 30) break;
            }
            if(finishedBeam.empty()) return 99999;
            
            std::vector<SearchNode> returnBeam;
            for(const auto& node : finishedBeam) {
                Coordinate bestSlot = findBestReturnSlot(node.yard, targetId, priorityMap, i, pm);
                if(bestSlot.row != -1) {
                    SearchNode rn = node;
                    rn.yard.initBox(targetId, bestSlot.row, bestSlot.bay, bestSlot.tier);
                    rn.f = rn.g;
                    returnBeam.push_back(rn);
                }
            }
            if(returnBeam.empty()) return 99999;
            currentBeam = returnBeam;
         }
         return currentBeam.empty() ? 99999 : currentBeam[0].g;
    }
};


// GA Module

class GeneticAlgorithm {
    struct Individual {
        std::vector<int> sequence;
        int fitness;
    };
    std::vector<Individual> population;
    YardSystem yardRef;
    std::mt19937 rng;
    PheromoneManager* pm; // Pointer to the 4D pheromone matrix

public:
    // Updated constructor to accept PheromoneManager
    GeneticAlgorithm(const YardSystem& yard, const std::vector<int>& targets, PheromoneManager* externalPM) 
        : yardRef(yard), pm(externalPM) {
        rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
        population.resize(POPULATION_SIZE);
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            population[i].sequence = targets;
            std::shuffle(population[i].sequence.begin(), population[i].sequence.end(), rng);
            population[i].fitness = std::numeric_limits<int>::max();
        }
    }

    void solve() {
        for (int gen = 0; gen < MAX_GENERATIONS; ++gen) {
            // 1. Parallel Fitness Calculation using OpenMP
            // Each thread calls BBS_Evaluator::evaluate with the pheromone matrix pointer
            #pragma omp parallel for
            for (int i = 0; i < POPULATION_SIZE; ++i) {
                if (population[i].fitness == std::numeric_limits<int>::max()) {
                    // Pass pm to guide the internal BBS search with learned pheromones
                    population[i].fitness = BBS_Evaluator::evaluate(yardRef, population[i].sequence, pm);
                }
            }
            
            // 2. Sort by fitness (Ascending order)
            std::sort(population.begin(), population.end(), [](const Individual& a, const Individual& b){ 
                return a.fitness < b.fitness; 
            });
            
            // 3. Pheromone Update (Learning Phase)
            // Every 10 generations, use the best individual to reinforce successful paths
            if (gen % 10 == 0 && population[0].fitness < 99999) {
                // Generate detailed logs for the best sequence to identify successful coordinates
                std::vector<MissionLog> bestLogs = BBS_Evaluator::solveAndRecord(yardRef, population[0].sequence, pm);
                pm->update(bestLogs, population[0].fitness);
            }
            
            // Progress report
            if (gen % 10 == 0 || gen == MAX_GENERATIONS - 1) {
                std::cout << "Gen " << std::setw(3) << gen << " | Best Cost: " << population[0].fitness << std::endl;
            }
            
            // 4. Evolution Logic (Elitism + Selection + Mutation)
            std::vector<Individual> nextGen;
            int eliteCount = POPULATION_SIZE * 0.1; 
            if (eliteCount < 1) eliteCount = 1;
            for(int i=0; i<eliteCount; ++i) nextGen.push_back(population[i]); // Keep elites
            
            while(nextGen.size() < POPULATION_SIZE) {
                // Tournament Selection (Simple half-population pool)
                const auto& p1 = population[std::uniform_int_distribution<int>(0, POPULATION_SIZE/2)(rng)];
                Individual child = p1;
                
                // Mutation: Swap two random containers in the sequence
                if(std::uniform_real_distribution<double>(0,1)(rng) < MUTATION_RATE) {
                    int idx1 = std::uniform_int_distribution<int>(0, child.sequence.size()-1)(rng);
                    int idx2 = std::uniform_int_distribution<int>(0, child.sequence.size()-1)(rng);
                    std::swap(child.sequence[idx1], child.sequence[idx2]);
                    child.fitness = std::numeric_limits<int>::max(); // Mark for re-evaluation
                }
                nextGen.push_back(child);
            }
            population = nextGen;
        }
    }

    std::vector<int> getBestSequence() { return population[0].sequence; }
    int getBestFitness() { return population[0].fitness; }
};


// Main Function

int main() {
    auto totalStart = std::chrono::high_resolution_clock::now();

    std::cout << "[Step 0] Loading Configuration..." << std::endl;
    YardConfig config = DataLoader::loadYardConfig("yard_config.csv");
    
    if (config.max_row == 0) {
        std::cout << "Using fallback defaults: 6x11x8, 400 boxes." << std::endl;
        config = {6, 11, 8, 400};
    } else {
        std::cout << "Config Loaded: " << config.max_row << "x" << config.max_bay 
                  << "x" << config.max_level << ", Capacity: " << config.total_boxes << std::endl;
    }

    // 1. Load Yard Layout
    std::cout << "[Step 1] Loading Yard Snapshot..." << std::endl;
    auto yardData = DataLoader::loadYardSnapshot("mock_yard.csv");
    if (yardData.empty()) { std::cerr << "Error: mock_yard.csv missing." << std::endl; return -1; }
    
    YardSystem yard(config.max_row, config.max_bay, config.max_level, config.total_boxes);
    for (const auto& box : yardData) yard.initBox(box.container_id, box.row, box.bay, box.level);

    // 2. Load Missions
    auto commandData = DataLoader::loadCommands("mock_commands.csv");
    if (commandData.empty()) { std::cerr << "Error: mock_commands.csv missing." << std::endl; return -1; }

    std::vector<int> targetBlockIds;
    std::vector<int> originalPrioritySeq;
    for (const auto& cmd : commandData) {
        if (cmd.cmd_type == "target" && yard.getBoxPosition(cmd.parent_carrier_id).row != -1) {
            targetBlockIds.push_back(cmd.parent_carrier_id);
            originalPrioritySeq.push_back(cmd.parent_carrier_id);
        }
    }
    if (targetBlockIds.empty()) { std::cerr << "Error: No valid targets." << std::endl; return -1; }

    // --- ACO INITIALIZATION ---
    // Initialize 4D Pheromone Matrix [Row][Bay][Level][Step_Index]
    // max_steps is capped (e.g., 50) to manage memory and learn periodic patterns
    PheromoneManager pm(config.max_row, config.max_bay, config.max_level, 50);

    // 3. Baseline Evaluation
    std::cout << "\n[Step 2] Calculating Original Sequence Cost..." << std::endl;
    // Initial evaluation without pheromone guidance
    int originalCost = BBS_Evaluator::evaluate(yard, originalPrioritySeq, &pm);
    std::cout << "Original Cost: " << originalCost << std::endl;

    // 4. GA Optimization with ACO Learning
    std::cout << "\n[Step 3] Running Hybrid GA-ACO Optimization..." << std::endl;
    auto gaStart = std::chrono::high_resolution_clock::now();
    
    // Pass the pheromone manager to GA for learning and guidance
    GeneticAlgorithm ga(yard, targetBlockIds, &pm);
    ga.solve();
    
    auto gaEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gaTime = gaEnd - gaStart;

    // 5. Compile Results
    std::vector<int> bestSeq = ga.getBestSequence();
    int bestCost = ga.getBestFitness();

    // 6. Generate Detailed Mission Logs (Final pass with learned pheromones)
    std::cout << "\n[Step 4] Generating Execution Logs..." << std::endl;
    std::vector<MissionLog> logs = BBS_Evaluator::solveAndRecord(yard, bestSeq, &pm);

    std::ofstream outFile("output_missions.csv");
    outFile << "mission_no,mission_type,batch_id,parent_carrier_id,source_position,dest_position,mission_priority,mission_status,created_time\n";
    for (const auto& m : logs) {
        std::stringstream ssSrc, ssDst;
        if (m.src.row == -1) ssSrc << "work station";
        else ssSrc << "(" << m.src.row << ";" << m.src.bay << ";" << m.src.tier << ")";
        if (m.dst.row == -1) ssDst << "work station";
        else ssDst << "(" << m.dst.row << ";" << m.dst.bay << ";" << m.dst.tier << ")";

        outFile << m.mission_no << "," << m.mission_type << "," << m.batch_id << "," << m.container_id << ","
                << ssSrc.str() << "," << ssDst.str() << "," << m.mission_priority << "," << m.mission_status << "," << m.created_time << "\n";
    }
    outFile.close();

    auto totalEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> totalTime = totalEnd - totalStart;

    // Report results
    std::cout << "\n================ EXPERIMENT REPORT ================" << std::endl;
    std::cout << "Optimization Time  : " << gaTime.count() << " sec" << std::endl;
    std::cout << "Total Elapsed Time : " << totalTime.count() << " sec" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Original Cost      : " << originalCost << std::endl;
    std::cout << "Optimized Cost     : " << bestCost << std::endl;
    double improvement = (originalCost > 0) ? (double)(originalCost - bestCost) / originalCost * 100.0 : 0;
    std::cout << "Improvement        : " << std::fixed << std::setprecision(2) << improvement << "%" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Detailed log saved to 'output_missions.csv'" << std::endl;

    return 0;
}
