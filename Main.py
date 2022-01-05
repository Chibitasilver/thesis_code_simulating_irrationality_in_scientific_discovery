from Machine import Machine
from Agent import Agent
import random
import matplotlib.pyplot as plt
import numpy as np


# Main environment

if __name__ == "__main__":

    # -------------------Model Parameters_________________

    # Internal limiters
    ACCEPTANCE_THRESHOLD = 0.90
    COMPLEXITY_LIMIT = 7
    DATA_LIMIT = 25

    # Finite State Machine dictionary
    alphabet = ['a', 'b', 'c']

    # Data, Finite State Machines and agents
    data_pool = []
    fst_pool  = []
    agents_pool = []

    # Variables to be changed for experimental setup
    NMR_OF_DATA_POINTS = 50  # 500
    NMR_OF_FSTS = 5  # 100
    NMR_OF_AGENTS = 1  # 60

    # __________________________________________________________________________________________________________________

    # Initialize FST pool

    for i in range(0, NMR_OF_FSTS):
        fst_pool.append(Machine(alphabet))
        fst_pool[i].define_random_fsm()

    # Initialize agent pool

    bias_list = np.round(np.random.uniform(0, 1, NMR_OF_AGENTS), decimals=3)

    for i in range(0, NMR_OF_AGENTS):
        agent = Agent(alphabet, bias_list[i])
        agents_pool.append(agent)

    agents_pool.sort(key=lambda x: x.BIAS)

    # Run Experimental setup

    biases = []
    structure_similarities = []
    results = []

    percentage = 0
    count = 0
    progress_bar_1 = ["["]
    for a in agents_pool:
        progress_bar_1.append(" ")
    progress_bar_2 = "]"
    bar_string = ""

    for agent in agents_pool:
        # Changed: Now iterate new data pool for every agent
        data_pool = agent.create_data(NMR_OF_DATA_POINTS, DATA_LIMIT)

        # Progress bar, works fine, don't touch
        for i in progress_bar_1:
            bar_string = bar_string + i
        print(bar_string + progress_bar_2 + " " + str(percentage) + "%")
        progress_bar_1[count + 1] = "*"
        count = count + 1
        percentage = (count / len(agents_pool)) * 100
        bar_string = ""

        for fst in fst_pool:
            while agent.model_not_done(COMPLEXITY_LIMIT, ACCEPTANCE_THRESHOLD):
                agent.build_model(fst, data_pool, COMPLEXITY_LIMIT)
            agent.results_pool.append(agent.accuracy)
            agent.agent_machine.show_fsm()
            fst.show_fsm()
            #agent.structure_pool.append((agent.check_structure_similarity(fst)))
            agent.reset()

        # Summarise results
        biases.append(agent.BIAS)
        results.append(np.mean(agent.results_pool))
        structure_similarities.append(np.mean(agent.structure_pool))

    print("")
    print("Done!")

    # Display results

    print("Bias: "  + str(biases))
    print("Accuracy: " + str(results))
    print("Structure similarity: " + str(structure_similarities))
    plt.bar(biases, results, width=0.01)
    plt.suptitle("bias factor vs. averaged model accuracy")
    plt.xlim(0, 1)
    plt.xlabel("Bias factor")
    plt.ylim(0, 1)
    plt.ylabel("Average prediction accuracy")
    plt.show()