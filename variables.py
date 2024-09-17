# https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)

# HYPERPARAMETERS
number_of_fold = 6
sampling = 0.2
stochastic = False
steps = 2000
# learning_rate = 4
learning_rate = 0.4
scheduler = None
scheduler_type = ["exp", "linear", None]
learning_rate_decay = 0.001


# FEATURES
labels_column = "Hogwarts House"

selected_features = [
    "Ancient Runes",
    "Astronomy",
    "Charms",
    "Defense Against the Dark Arts",
    "Divination",
    "Flying",
    "Herbology",
]

unselected_features = [
    "Arithmancy",
    "Care of Magical Creatures",
    "History of Magic",
    "Muggle Studies",
    "Potions",
    "Transfiguration",
]

histogram_feature = "Care of Magical Creatures"
scatter_feature_pair = "Arithmancy - Care of Magical Creatures"


# COLORS
blue = "#83a598"
green = "#b8bb26"
yellow = "#fabd2f"
red = "#fb4934"
purple = "#b16286"

colors = {
    "Ravenclaw": blue,
    "Slytherin": green,
    "Hufflepuff": yellow,
    "Gryffindor": red,
    "unknown": purple,
}
