import java.util.*;

// Node class for Huffman tree
class HuffmanNode {
    char ch;
    int freq;
    HuffmanNode left, right;

    HuffmanNode(char ch, int freq) {
        this.ch = ch;
        this.freq = freq;
        this.left = null;
        this.right = null;
    }
}

// Comparator for priority queue (min-heap)
class CompareNode implements Comparator<HuffmanNode> {
    public int compare(HuffmanNode x, HuffmanNode y) {
        return x.freq - y.freq;
    }
}

public class HuffmanCoding {

    // Recursive function to print Huffman codes
    public static void printCodes(HuffmanNode root, String code) {
        if (root == null)
            return;

        // If it's a leaf node, print the character and its code
        if (root.left == null && root.right == null && Character.isLetter(root.ch)) {
            System.out.println(root.ch + " : " + code);
            return;
        }

        printCodes(root.left, code + "0");
        printCodes(root.right, code + "1");
    }

    public static void main(String[] args) {
        // Example characters and frequencies
        char[] chars = { 'A', 'B', 'C', 'D', 'E', 'F' };
        int[] freqs = { 5, 9, 12, 13, 16, 45 };

        // Create a priority queue (min-heap)
        PriorityQueue<HuffmanNode> pq = new PriorityQueue<>(new CompareNode());

        // Step 1: Create leaf nodes for each character
        for (int i = 0; i < chars.length; i++) {
            pq.add(new HuffmanNode(chars[i], freqs[i]));
        }

        // Step 2: Build the Huffman tree
        while (pq.size() > 1) {
            // Extract two smallest frequency nodes
            HuffmanNode left = pq.poll();
            HuffmanNode right = pq.poll();

            // Create a new internal node with sum of frequencies
            HuffmanNode newNode = new HuffmanNode('$', left.freq + right.freq);
            newNode.left = left;
            newNode.right = right;

            // Insert new node back to queue
            pq.add(newNode);
        }

        // Step 3: Print Huffman codes from the root
        HuffmanNode root = pq.peek();
        System.out.println("Huffman Codes:");
        printCodes(root, "");
    }
}
