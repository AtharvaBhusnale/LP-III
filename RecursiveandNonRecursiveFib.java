public class RecursiveandNonRecursiveFib {

    // Recursive method to find Fibonacci number
    public static int fibonacciRecursive(int n) {
        if (n <= 1) {
            return n;
        }
        return fibonacciRecursive(n - 1) + fibonacciRecursive(n - 2);
    }

    // Non-recursive (iterative) method to find Fibonacci number
    public static int fibonacciNonRecursive(int n) {
        if (n <= 1) {
            return n;
        }
        int prev = 0,
            curr = 1;
        for (int i = 2; i <= n; i++) {
            int next = prev + curr;
            prev = curr;
            curr = next;
        }
        return curr;
    }

    public static void main(String[] args) {
        int n = 10;
        System.out.println(
            "Fibonacci number at position " +
                n +
                " (Recursive): " +
                fibonacciRecursive(n)
        );
        System.out.println(
            "Fibonacci number at position " +
                n +
                " (Non-Recursive): " +
                fibonacciNonRecursive(n)
        );
    }
}
