# https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)

# HYPERPARAMETERS
number_of_fold = 5
sampling = 0.2
stochastic = False
steps = 1000
# learning_rate = 4
learning_rate = 0.2
scheduler = None
learning_rate_decay = 0


# FEATURES
labels_column = "Hogwarts House"

selected_features = [
    "Ancient Runes",
    "Astronomy",
    "Charms",
    "Divination",
    "Flying",
    "Defense Against the Dark Arts",
    "Muggle Studies",
    "Transfiguration",
    "History of Magic",
    "Herbology",
]

unselected_features = [
    "Arithmancy",
    "Care of Magical Creatures",
    "Potions",
]


# FILES
prediction_file = "houses.csv"


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
    "Test_Set": purple,
}
