public class NQueen {
    static int N; // size of the board

    // Function to print the chessboard configuration
    static void printSolution(int[][] board) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                System.out.print(board[i][j] == 1 ? "Q " : ". ");
            }
            System.out.println();
        }
        System.out.println();
    }

    // Function to check if a queen can be placed at board[row][col]
    static boolean isSafe(int[][] board, int row, int col) {
        int i, j;

        // Check this column on the upper side
        for (i = 0; i < row; i++)
            if (board[i][col] == 1)
                return false;

        // Check upper left diagonal
        for (i = row, j = col; i >= 0 && j >= 0; i--, j--)
            if (board[i][j] == 1)
                return false;

        // Check upper right diagonal
        for (i = row, j = col; i >= 0 && j < N; i--, j++)
            if (board[i][j] == 1)
                return false;

        return true;
    }

    // Recursive function to solve N Queen problem
    static boolean solveNQUtil(int[][] board, int row) {
        // Base case: all queens placed
        if (row >= N)
            return true;

        // Try placing a queen in all columns of the current row
        for (int col = 0; col < N; col++) {
            if (isSafe(board, row, col)) {
                board[row][col] = 1; // place queen

                // Recur to place the rest of the queens
                if (solveNQUtil(board, row + 1))
                    return true;

                // Backtrack: remove the queen
                board[row][col] = 0;
            }
        }
        return false; // no valid place in this row
    }

    // Main function to solve the N Queen problem
    static boolean solveNQ(int n) {
        N = n;
        int[][] board = new int[N][N];

        if (!solveNQUtil(board, 0)) {
            System.out.println("No solution exists for N = " + N);
            return false;
        }

        printSolution(board);
        return true;
    }

    // Driver code
    public static void main(String[] args) {
        int n = 8; // You can change this value to test different N
        solveNQ(n);
    }
}