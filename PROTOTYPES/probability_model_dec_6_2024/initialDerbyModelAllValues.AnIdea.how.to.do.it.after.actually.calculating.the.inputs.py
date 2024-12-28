# this script is just for inspiration of how i might do probabilities after i have calculated them all ? 

# right now it is more of a mock up idea but a start ! 
def predict_stock_movement(streak_prob, stdev_prob, deriv_prob, interest_rate_prob, spy_prob, other_probs, weights):
    """Calculates the probability of a stock going up based on various factors.

    Args:
        streak_prob: Probability of going up based on streak length.
        stdev_prob: Probability of going up based on standard deviations from the mean.
        deriv_prob: Probability of going up based on the second derivative of price.
        interest_rate_prob: Probability of going up based on interest rates.
        spy_prob: Probability of going up based on SPY movement.
        other_probs: List of probabilities of going up based on other factors.
        weights: List of weights corresponding to each probability.

    Returns:
        A tuple (prob_up, prob_down) representing the probabilities of the stock
        going up and down, respectively.
    """

    # Combine all probabilities
    all_probs = [streak_prob, stdev_prob, deriv_prob, interest_rate_prob, spy_prob] + other_probs

    # Calculate weighted average
    prob_up = sum(prob * weight for prob, weight in zip(all_probs, weights))
    prob_down = 1 - prob_up

    return prob_up, prob_down

# Example usage
if __name__ == "__main__":
    streak_prob = 0.45  # Replace with your calculated value
    stdev_prob = 0.60
    deriv_prob = 0.75
    interest_rate_prob = 0.55
    spy_prob = 0.68

    # Add probabilities for other factors (if any)
    other_probs = [0.58, 0.62]  # Example: two additional factors

    # Make sure the weights sum to 1
    weights = [0.20, 0.15, 0.25, 0.10, 0.30] + [0.10, 0.10] # Adjust weights based on your analysis

    prob_up, prob_down = predict_stock_movement(streak_prob, stdev_prob, deriv_prob, interest_rate_prob, spy_prob, other_probs, weights)

    print(f"Probability of stock going up: {prob_up:.2%}")
    print(f"Probability of stock going down: {prob_down:.2%}")

