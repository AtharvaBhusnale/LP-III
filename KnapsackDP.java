public class KnapsackDP {
    
    // Function to solve 0/1 Knapsack problem
    public static int knapSack(int W, int[] wt, int[] val, int n) {
        int[][] dp = new int[n + 1][W + 1];

        // Build table dp[][] in bottom-up manner
        for (int i = 0; i <= n; i++) {
            for (int w = 0; w <= W; w++) {
                if (i == 0 || w == 0)
                    dp[i][w] = 0;
                else if (wt[i - 1] <= w)
                    dp[i][w] = Math.max(val[i - 1] + dp[i - 1][w - wt[i - 1]], dp[i - 1][w]);
                else
                    dp[i][w] = dp[i - 1][w];
            }
        }
        return dp[n][W]; // Maximum value that can be put in the knapsack
    }

    // Driver code
    public static void main(String[] args) {
        int[] val = {60, 100, 120}; // values (profits)
        int[] wt = {10, 20, 30}; // weights
        int W = 50; // capacity of knapsack
        int n = val.length;
        
        System.out.println("Maximum value in Knapsack = " + knapSack(W, wt, val, n));
    }
}