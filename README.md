# Discussion Assignment 1

My intuitive guess was that the Blue Crescent and Blue Arrow would fall into Group 1 ("Yes") and the yellow ring would fall into Group 0 ("No") based on the fact that we see no blue toys in Group 0, nor any yellow toys in group 1.

However, given the instruction that we want to attempt to assess this from a probabilistic perspective rather than an intuitive one, I wanted to establish some kind of metric that could justify the intuition that the blue crescent would fall into Group 1, and then use that metric to automate the classification of the other two toys. This eventually led me to getting familiar with the use of the Naive Bayes formula and the use of Laplace Smoothing, which I understand is a tool to deal with ambiguity and promote generalization. The use of this metric produced classifications aligned with my intuitions:

```Results
Object 1: ('Blue', 'Crescent', 'Larger')
Probabilities: Yes = 0.7260, No = 0.2740
Classified as: Yes

Object 2: ('Yellow', 'Ring', 'Larger')
Probabilities: Yes = 0.1283, No = 0.8717
Classified as: No

Object 3: ('Blue', 'Arrow', 'Larger')
Probabilities: Yes = 0.5699, No = 0.4301
Classified as: Yes
```

### Defining Problem Space / Setting Assumptions: 

I figured that since there are three features we can observe for any given shape, we could use the frequency at which these features appears in both among toys in the 'Yes' and 'No' categories respectively to derive some sort of probability-based scoring system for our unknown cases.

For the Blue Crescent, it's features would be as follows:
- Shape: Crescent
- Color: Blue
- Size: "Larger"

For defining the feature of size, It was immediately clear to me that using some kind of exact description would be challenging, given that we are only given the exact sizes for three toys, without any clarification of units (surface area? Perimeter?) and the three given values did not seem to obviously correlate to what could be visually inspected. Thus I chose to make binary classification of size for each toy -- "larger," or "smaller." In this Approximation, anything that is known or appears to be =<10 units of size is classified as 'Larger':

**Group 1 ("Yes")**
- Square, Blue, Larger
- Ellipse, Red, Smaller
- Ring, Blue, Larger
- Circle, Green, Smaller
- Circle, Orange, Smaller
- Circle, Orange, Smaller
- Rectangle, Blue, Larger
- 4-Point Star, Blue, Larger
- Circle, Blue, Smaller

**Group 0: ("No"):**
- 5-Point Star, Yellow, Larger
- Arrow, Red, Larger
- Ring, Red, Larger
- Parallelogram, Green, Larger
- Triangle, Yellow, Larger
- Parallelogram, Green, Larger
- Ellipse, Orange, Larger
- Ellipse, Red, Larger

**Unclassified Group ("???")**
- Total Toys: 3
	- 1: Blue, Crescent, Larger
	- 2: Yellow, Ring, Larger
	- 3: Blue, Arrow, Larger

### Counting Features:

I used a simple Python script to verify my manual count of the frequency of all features (credit to ChatGPT-4 for these scripts --which indeed caught an error in my counting): 

```Python
# Let's verify the counts by structuring a simple Python script.

# First, define the toy descriptions for each group
group_yes_descriptions = [
    ("Square", "Blue", "Larger"),
    ("Ellipse", "Red", "Smaller"),
    ("Ring", "Blue", "Larger"),
    ("Circle", "Green", "Smaller"),
    ("Circle", "Orange", "Smaller"),
    ("Circle", "Orange", "Smaller"),
    ("Rectangle", "Blue", "Larger"),
    ("4-Point Star", "Blue", "Larger"),
    ("Circle", "Blue", "Smaller")
]

group_no_descriptions = [
    ("5-Point Star", "Yellow", "Larger"),
    ("Arrow", "Red", "Larger"),
    ("Ring", "Red", "Larger"),
    ("Parallelogram", "Green", "Larger"),
    ("Triangle", "Yellow", "Larger"),
    ("Parallelogram", "Green", "Larger"),
    ("Ellipse", "Orange", "Larger"),
    ("Ellipse", "Red", "Larger")
]

# Function to count the frequency of each feature in the given group
def count_features(group):
    color_count = {}
    shape_count = {}
    size_count = {'Smaller': 0, 'Larger': 0}
    
    for shape, color, size in group:
        color_count[color] = color_count.get(color, 0) + 1
        shape_count[shape] = shape_count.get(shape, 0) + 1
        size_count[size] += 1
    
    return color_count, shape_count, size_count

# Count features for both groups
group_yes_color_count, group_yes_shape_count, group_yes_size_count = count_features(group_yes_descriptions)
group_no_color_count, group_no_shape_count, group_no_size_count = count_features(group_no_descriptions)

(group_yes_color_count, group_yes_shape_count, group_yes_size_count), (group_no_color_count, group_no_shape_count, group_no_size_count)
```

```Result
(({'Blue': 5, 'Red': 1, 'Green': 1, 'Orange': 2},
  {'Square': 1,
   'Ellipse': 1,
   'Ring': 1,
   'Circle': 4,
   'Rectangle': 1,
   '4-Point Star': 1},
  {'Smaller': 5, 'Larger': 4}),
 ({'Yellow': 2, 'Red': 3, 'Green': 2, 'Orange': 1},
  {'5-Point Star': 1,
   'Arrow': 1,
   'Ring': 1,
   'Parallelogram': 2,
   'Triangle': 1,
   'Ellipse': 2},
  {'Smaller': 0, 'Larger': 8}))
```


### Using Naive Bayes and Laplace Smoothing

#### 1: Count Frequency of the Features of Crescent, Blue, Medium

The first thing I chose to do was break down the features of the Blue, Larger Crescent and consider how frequently each of these features appeared within the existing dataset:

| Feature  | Total (Yes & No) | Yes           | No            |
| -------- | ---------------- | ------------- | ------------- |
| Blue     | 5                | 5/5 = [1.0]   | 0/5 = [0.0]   |
| Crescent | 0                | 0/0 = [n/a?]  | 0/0 = [n/a?]  |
| Larger   | 12               | 4/12 = [0.33] | 8/12 = [0.66] |

At this point, I got a bit confused by how to handle things such as the frequency of 0 for blue toys in Group 0 ("No"), and even more confused about how to handle the Crescent shape which fails to appear in either group. What I could determine is that it's fair to assume all three features are independent of one another, and that they are not mutually exclusive in any obvious way. 

Conversing with ChatGPT-4 about this, it pointed me towards the use of the Naive Bayes formula, and the specific use of Laplace Smoothing to deal with the probabilities of 0 or undefined values, which could otherwise hinder the ability of my metric to generalize to unseen features and combinations of features.

#### 2: Apply Laplace Smoothing for 'Crescent' feature to account for it being unobserved in all Groups. 

The formula for Laplace Smoothing, as explained by ChatGPT looks as follows: 

$$ P(f_i | c) = \frac{ \text{count of } f_i \text{ in class } c + 1}{\text{total count of class } c + \text{number of possible values of } f_i} $$

I interpreted it to be used as follow in order to establish a probability for the yet-to-be-seen crescent shape in both Group 1 (Yes) and Group 0 (No):

**For Yes:**
 >$P(\text{Crescent} | \text{Yes}) = \frac{\text{Count of Crescent in Class Yes} + 1}{\text{Total Count of Toys in Class Yes} + \text{Count of Unique Shapes in Both Classes}}$
 >
 >$P(\text{Crescent} | \text{Yes}) = \frac{0 + 1}{\text{9} + 10}$
 >
 >$P(\text{Crescent} | \text{Yes}) = 0.0526$
 >

**For No:**
 >$P(\text{Crescent} | \text{No}) = \frac{\text{Count of Crescent in Class No} + 1}{\text{Total Count of Toys in Class No} + \text{Count of Unique Shapes in Both Classes}}$
 >
 >$P(\text{Crescent} | \text{No}) = \frac{0 + 1}{\text{8} + 10}$
 >
 >$P(\text{Crescent} | \text{No}) = 0.0556$

#### 3: UPDATED Frequencies w/ Laplace Smoothing applied to all Values

GPT-4 also advised using Laplace Smoothing in calculating probabilities for all features, as a means of promoting robustness and generalization in the Naive Bayes method. I can't be sure if it was correct in this advice (and I look forward to learning if it is) -- but decided to follow it's suggestion anyways, giving me the following probabilities:  

| Feature  | Yes    | No     |
| -------- | ------ | ------ |
| Blue     | 0.4615 | 0.0833 |
| Crescent | 0.0526 | 0.0556 |
| Larger   | 0.4545 | 0.9000 |
| Prior    | 0.5    | 0.5    |
For the priors, because the total number of classified toys is an odd number of 17, it's impossible to have  50/50 split. But because we do not see any skewed distribution of the already classified values, such as Yes = 10, No =7, a split of 8/9 is as close as possible to 50/50. Therefore I decided to assume a prior of P(Yes) = P(No) = 0.5. GPT-4 also advised that the use of 50/50 priors is good practice when no data suggests otherwise. 

#### 4: Calculate Probabilities using Naive Bayes Formula:

The formula for Naive Bayes was conveyed to me by GPT-4 as follow:
$$P(c | f_1, f_2, \ldots, f_n) \approx P(c) \prod_{i=1}^{n} P(f_i | c)$$

I interpreted that it should be applied as follows:

**For Yes:**
>$P(\text{Yes} | f_{Blue}, f_{Crescent}, f_{Larger}) \approx P(\text{Yes}) \times P(f_{Blue} | \text{Yes}) \times P(f_{Crescent} | \text{Yes}) \times P(f_{Larger} | \text{Yes})$
>
>$P(\text{Yes} | f_{Blue}, f_{Crescent}, f_{Larger}) \approx 0.5 \times 0.4615 \times 0.05263 \times 0.4545$
>
>$P(\text{Yes} | f_{Blue}, f_{Crescent}, f_{Larger}) \approx 0.0055$


**For No:**
>$P(\text{No} | f_{Blue}, f_{Crescent}, f_{Larger}) \approx P(\text{No}) \times P(f_{Blue} | \text{No}) \times P(f_{Crescent} | \text{No}) \times P(f_{Larger} | \text{No})$
>
>$P(\text{No} | f_{Blue}, f_{Crescent}, f_{Larger}) \approx 0.5 \times 0.0833 \times 0.0555 \times 0.9000$
>
>$P(\text{No} | f_{Blue}, f_{Crescent}, f_{Larger}) \approx 0.0021$


And then some simple arithmetic for **Normalization** as follows: 

$Total = 0.0055 + 0.0021 = 0.0076$
$0.0055 / 0.0076 = 0.7258$
$0.0021 / 0.0076 = 0.2742$

My attempt led me to conclude that Blue Crescent has a 72% Probability of being in Group 1 ("Yes"), and a 27% Probability of being in Group 0 ("No"), according to the Naive Bayes method. This validated what intuition already suggested, and made me feel confident to apply this method to the two other unclassified toys. 
#### 5: Scripting the Process for all three Unclassified Cases:

A script thrown together to apply Laplace Smoothing and the Naive Bayes method to all three unclassified toys, using the established toy groupings to define the probability inputs for Naive Bayes. Once again, credit to ChatGPT for generating this script: 

```Python
# Define the classified and unclassified datasets
group_yes_descriptions = [
    ("Square", "Blue", "Larger"),
    ("Ellipse", "Red", "Smaller"),
    ("Ring", "Blue", "Larger"),
    ("Circle", "Green", "Smaller"),
    ("Circle", "Orange", "Smaller"),
    ("Circle", "Orange", "Smaller"),
    ("Rectangle", "Blue", "Larger"),
    ("4-Point Star", "Blue", "Larger"),
    ("Circle", "Blue", "Smaller")
]

group_no_descriptions = [
    ("5-Point Star", "Yellow", "Larger"),
    ("Arrow", "Red", "Larger"),
    ("Ring", "Red", "Larger"),
    ("Parallelogram", "Green", "Larger"),
    ("Triangle", "Yellow", "Larger"),
    ("Parallelogram", "Green", "Larger"),
    ("Ellipse", "Orange", "Larger"),
    ("Ellipse", "Red", "Larger")
]

unclassified_toys = [
    ("Blue", "Crescent", "Larger"),
    ("Yellow", "Ring", "Larger"),
    ("Blue", "Arrow", "Larger")
]

# Helper function to count the frequency of features
def count_features(group):
    color_count, shape_count, size_count = {}, {}, {'Smaller': 0, 'Larger': 0}
    for shape, color, size in group:
        color_count[color] = color_count.get(color, 0) + 1
        shape_count[shape] = shape_count.get(shape, 0) + 1
        size_count[size] += 1
    return color_count, shape_count, size_count

# Count features in classified groups
yes_color_count, yes_shape_count, yes_size_count = count_features(group_yes_descriptions)
no_color_count, no_shape_count, no_size_count = count_features(group_no_descriptions)

# Total toys in each group
total_yes = len(group_yes_descriptions)
total_no = len(group_no_descriptions)

# Unique shapes across both groups
unique_shapes = len(set(list(yes_shape_count.keys()) + list(no_shape_count.keys())))

# Function to calculate probabilities using Naive Bayes with Laplace smoothing
def calculate_probabilities(toy):
    color, shape, size = toy
    results = []
    for group_name, (color_count, shape_count, size_count, total) in [
        ("Yes", (yes_color_count, yes_shape_count, yes_size_count, total_yes)),
        ("No", (no_color_count, no_shape_count, no_size_count, total_no))
    ]:
        # Apply Laplace smoothing for color and size (shape is explicitly smoothed later)
        p_color = (color_count.get(color, 0) + 1) / (total + len(color_count))
        p_shape = (shape_count.get(shape, 0) + 1) / (total + unique_shapes) # Laplace smoothing for shape
        p_size = (size_count[size] + 1) / (total + len(size_count)) # Assuming only two sizes, so +2 in denominator
        
        # Calculate the final probability for the group
        p_final = p_color * p_shape * p_size
        results.append((group_name, p_final))
    return results

# Iterate over unclassified toys and calculate probabilities
for i, toy in enumerate(unclassified_toys, 1):
    probabilities = calculate_probabilities(toy)
    yes_prob = next(p for g, p in probabilities if g == "Yes")
    no_prob = next(p for g, p in probabilities if g == "No")
    
    # Normalize probabilities
    total_prob = yes_prob + no_prob
    normalized_yes_prob = yes_prob / total_prob
    normalized_no_prob = no_prob / total_prob
    
    # Decide on the class based on higher probability
    classification = "Yes" if normalized_yes_prob > normalized_no_prob else "No"
    
    print(f"Object {i}: {toy}")
    print(f"Probabilities: Yes = {normalized_yes_prob:.4f}, No = {normalized_no_prob:.4f}")
    print(f"Classified as: {classification}\n")
```

```Results
Object 1: ('Blue', 'Crescent', 'Larger')
Probabilities: Yes = 0.7260, No = 0.2740
Classified as: Yes

Object 2: ('Yellow', 'Ring', 'Larger')
Probabilities: Yes = 0.1283, No = 0.8717
Classified as: No

Object 3: ('Blue', 'Arrow', 'Larger')
Probabilities: Yes = 0.5699, No = 0.4301
Classified as: Yes
```





