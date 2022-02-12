from Machine import Machine
from Agent import Agent
import matplotlib.pyplot as plt
import numpy as np

# Main environment

if __name__ == "__main__":

    # -------------------Model Parameters_________________

    # Internal limiters
    ACCEPTANCE_THRESHOLD = 0.75
    COMPLEXITY_LIMIT = 7
    DATA_LIMIT = 25

    # Finite State Machine dictionary
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f']

    # Data, Finite State Machines and agents
    data_pool = []
    fst_pool = []
    agents_pool = []

    # Variables to be changed for experimental setup
    NMR_OF_DATA_POINTS = 50  # 50
    NMR_OF_FSTS = 30  # 30
    NMR_OF_AGENTS = 100  # 100

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

    accuracies = []
    structure_similarities = []
    sizes = []
    nr_of_transitions = []

    data_std = []
    structure_std = []
    size_std = []
    transitions_std = []

    # Percentage bar

    percentage = 0
    count = 0
    progress_bar_1 = ["["]
    for a in agents_pool:
        progress_bar_1.append(" ")
    progress_bar_2 = "]"
    bar_string = ""

    # Agent training

    for agent in agents_pool:
        data_pool = agent.create_data(NMR_OF_DATA_POINTS, DATA_LIMIT)

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

            # Add agent results to relevant result pools

            agent.accuracy_pool.append(agent.accuracy)
            agent.structure_pool.append((agent.check_structure_similarity(fst)))
            agent.size_pool.append(len(agent.agent_machine.transition_table) == len(fst.transition_table))
            agent.nr_of_transitions_pool.append(agent.number_of_transitions())
            agent.reset()

        # Summarise results

        biases.append(agent.BIAS)
        accuracies.append(np.mean(agent.accuracy_pool))
        structure_similarities.append(np.mean(agent.structure_pool))
        sizes.append(sum(agent.size_pool) / len(agent.size_pool))
        nr_of_transitions.append((np.mean(agent.nr_of_transitions_pool)))

        data_std.append(np.std(agent.accuracy_pool))
        structure_std.append(np.std(agent.structure_pool))
        size_std.append(np.std(agent.size_pool))
        transitions_std.append(np.std(agent.nr_of_transitions_pool))

    print("")
    print("Done!")

    # Chance agent training

    chance_agent = Agent(alphabet, 0)

    for fst in fst_pool:
        chance_agent.agent_machine.define_random_fsm()
        chance_agent.accuracy_pool.append(chance_agent.accuracy)
        chance_agent.structure_pool.append(chance_agent.check_structure_similarity(fst))
        chance_agent.size_pool.append(chance_agent.agent_machine.transition_table == len(fst.transition_table))
        chance_agent.nr_of_transitions_pool.append(chance_agent.number_of_transitions())
        chance_agent.reset()

    chance_fit = np.mean(chance_agent.accuracy_pool)
    chance_size_match = sum(chance_agent.size_pool) / len(chance_agent.size_pool)
    chance_structure_match = np.mean(chance_agent.structure_pool)
    chance_nr_of_transitions = np.mean(chance_agent.nr_of_transitions_pool)

    # Display results

    fig, axs = plt.subplots(2, 2)

    # Irrationality vs fit on data

    axs[0, 0].plot(biases, accuracies, label='Scientific agent')
    axs[0, 0].fill_between(biases, np.array(accuracies) - np.array(data_std), np.array(accuracies) + np.array(data_std),
                           alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    axs[0, 0].plot(biases, np.full(len(biases), chance_fit), label='Chance level')
    axs[0, 0].set_title('Agent irrationality VS fit on data')
    axs[0, 0].set(xlabel=' Irrationality factor', ylabel='Average fit on data')
    axs[0, 0].set_xlim([0, 1])
    axs[0, 0].set_ylim([0, 1])
    axs[0, 0].legend()

    # Irrationality vs structure similarity match

    axs[0, 1].plot(biases, structure_similarities, label='Scientific agent')
    axs[0, 1].fill_between(biases, np.array(structure_similarities) - np.array(structure_std),
                           np.array(structure_similarities) + np.array(structure_std), alpha=0.5, edgecolor='#CC4F1B',
                           facecolor='#FF9848')
    axs[0, 1].plot(biases, np.full(len(biases), chance_structure_match), label='Chance level')
    axs[0, 1].set_title('Agent irrationality VS structure similarity match')
    axs[0, 1].set(xlabel=' Irrationality factor', ylabel='Average structure similarity')
    axs[0, 1].set_xlim([0, 1])
    axs[0, 1].set_ylim(bottom=0)
    axs[0, 1].legend()

    # Irrationality vs structure size match

    axs[1, 0].plot(biases, sizes, label='Scientific agent')
    axs[1, 0].fill_between(biases, np.array(sizes) - np.array(size_std), np.array(sizes) + np.array(size_std),
                           alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    axs[1, 0].plot(biases, np.full(len(biases), chance_size_match), label='Chance level')
    axs[1, 0].set_title('Agent irrationality VS structure size match')
    axs[1, 0].set(xlabel=' Irrationality factor', ylabel='% size match')
    axs[1, 0].set_xlim([0, 1])
    axs[1, 0].set_ylim(bottom=0)
    axs[1, 0].legend()

    # Irrationality vs nr_of_transitions

    axs[1, 1].plot(biases, nr_of_transitions, label='Scientific agent')
    axs[1, 1].fill_between(biases, np.array(nr_of_transitions) - np.array(transitions_std),
                           np.array(nr_of_transitions) + np.array(transitions_std), alpha=0.5, edgecolor='#CC4F1B',
                           facecolor='#FF9848')
    axs[1, 1].plot(biases, np.full(len(biases), chance_nr_of_transitions), label='Chance level')
    axs[1, 1].set_title('Agent irrationality VS average number of transitions')
    axs[1, 1].set(xlabel=' Irrationality factor', ylabel='Nr of transitions')
    axs[1, 1].set_xlim([0, 1])
    axs[1, 1].set_ylim(bottom=0)
    axs[1, 1].legend()

    fig.suptitle(
        "The irrationality factor of scientific agents and how it affects various aspects of their created theories")

    plt.show()
