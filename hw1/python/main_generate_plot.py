import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name0', '-xn0', type=str, default='my_exp', help='name of the experiment')
    parser.add_argument('--exp_name1', '-xn1', type=str, default='my_exp', help='name of the experiment')
    parser.add_argument('--exp_name2', '-xn2', type=str, default='my_exp', help='name of the experiment')
    args = parser.parse_args()
    params = vars(args)

    # Load pkl-file containing the learning (reward) history
    file_name1 = params['exp_name0'] + '.pkl'
    file_name2 = params['exp_name1'] + '.pkl'
    file_name3 = params['exp_name2'] + '.pkl'
    with open(file_name1, 'rb') as f:
        ro_reward1 = pickle.load(f)

    with open(file_name2, 'rb') as f:
        ro_reward2 = pickle.load(f)
            
    with open(file_name3, 'rb') as f:
        ro_reward3 = pickle.load(f)
    # Plot the data
    sns.lineplot(data=ro_reward1, linestyle='--', label='tr0')
    sns.lineplot(data=ro_reward2, linestyle='--', label='tr1')
    sns.lineplot(data=ro_reward3, linestyle='--', label='tr2')
    plt.xlabel('rollout', fontsize=25, labelpad=-2)
    plt.ylabel('reward', fontsize=25)
    plt.title('Learning curve for Cartpole', fontsize=30)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
