import numpy as np

def generate_ballots(votes=100, candidates=6, target_results=[30, 27, 21, 13, 6, 3]):
    '''
    Generate random ballot data given a total number of ballots (votes),
    a total number of candidates, and a target probability distribution
    of preferences.
    The probabilities are randomly attributed to each candidate for each
    level of preference.
    
    Returns a NumPy array with shape (votes, candidates), where each row
    is one ranked-choice ballot for one voter, and each column corresponds
    to one candidate.
    '''
    # Initialise a random number generator
    rng = np.random.default_rng()
    
    # Set target probabilities for each stage (normalised)
    prob = np.array(target_results, dtype=float)
    prob = np.tile(prob, (candidates, 1))
    # shuffle probabilities so they're applied differently for each rank; add some noise too
    prob = np.abs(rng.permuted(prob, axis=1) + rng.normal(scale=2, size=prob.shape))
    
    # Create an empty array to store the ballots
    ballots = np.zeros((votes, candidates))
    
    # Create each ballot one after the other
    for v in range(votes):
        # Voter ranks at most "candidates" candidates; introduce "stages" for clarity
        stages = candidates
        for r in range(stages):
            
            # Generates rth preference for an arbitrary candidate (use normalised probabilities)
            chosen_candidate = rng.choice(candidates, p=prob[r, :]/prob[r, :].sum())
            
            if ballots[v, chosen_candidate-1] > 0:
                # Arbitrarily decide that voter is done if they choose the same candidate twice
                break
            else:
                # If they hadn't previously chosen that candidate, choose it as rth preference (r indexes from 0)
                ballots[v, chosen_candidate-1] = r + 1
    
    return ballots


def select_ballots(ballots, rank, candidate):
    '''
    Returns a selector for all ballots which have allocated a given rank to a given candidate.
    '''
    # Create a bool mask: look at one candidate's column,
    # and find all the rows where that candidate's rank is rank
    return ballots[:, candidate] == rank

def tally_preferences(ballots, rank):
    
    # Determine the number of candidates based on the number of columns in the ballots array.
    candidates = ballots.shape[1]
    
    # Initialize a preference count array with zeros for each candidate.
    preferences = np.zeros(candidates, dtype=int)

    # Iterate over each candidate index.
    for candidate in range(candidates):
        # Count how many times the current candidate's rank matches the given rank across all ballots.
        preferences[candidate] = np.sum(ballots[:, candidate] == rank)

    # Return the tally of preferences for each candidate.
    return preferences

# Run this cell to test your function (don't change this code!)
importlib.reload(bd);
ballots = np.loadtxt('testing/tally.txt')

# Number of first preferences
preferences = bd.tally_preferences(ballots, 1)
np.testing.assert_equal(preferences, [4, 0, 4, 2])

# Number of third preferences
preferences = bd.tally_preferences(ballots, 3)
np.testing.assert_equal(preferences, [1, 1, 2, 0])

print('Passed the tests.')
