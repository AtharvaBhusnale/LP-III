import java.util.*;

// Item class to represent each object
class Item {
    int value;
    int weight;

    Item(int value, int weight) {
        this.value = value;
        this.weight = weight;
    }
}

public class FractionalKnapsack {

    // Comparator to sort items by value/weight ratio in descending order
    public static class ItemComparator implements Comparator<Item> {
        public int compare(Item a, Item b) {
            double r1 = (double) a.value / a.weight;
            double r2 = (double) b.value / b.weight;
            return Double.compare(r2, r1); // descending order
        }
    }

    // Function to return maximum total value in knapsack
    public static double getMaxValue(Item[] items, int capacity) {
        Arrays.sort(items, new ItemComparator());

        double totalValue = 0.0;

        for (Item item : items) {
            if (capacity == 0)
                break;

            if (item.weight <= capacity) {
                // Take the whole item
                capacity -= item.weight;
                totalValue += item.value;
            } else {
                // Take fraction of the item
                double fraction = (double) capacity / item.weight;
                totalValue += item.value * fraction;
                capacity = 0;
            }
        }
        return totalValue;
    }

    public static void main(String[] args) {
        Item[] items = {
            new Item(60, 10),
            new Item(100, 20),
            new Item(120, 30)
        };
        int capacity = 50;

        double maxValue = getMaxValue(items, capacity);
        System.out.println("Maximum value in the knapsack = " + maxValue);
    }
}