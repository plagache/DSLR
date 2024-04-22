# https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)

# HYPERPARAMETERS
# learning_rate = 1e-1
# learning_rate = 2
learning_rate = 4
# learning_rate = 8
# learning_rate_decay = 1e1
# learning_rate_decay = 1e-2
# learning_rate_decay = 0
learning_rate_decay = 1e-5
# learning_rate_decay = 1e-7
# learning_rate_decay = 1e-1
# steps = 50
# steps = 100
# steps = 600
steps = 1600
stochastic = False
# stochastic = True
scheduler = "exp"
sampling = 0


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

colors = {
    "Ravenclaw": blue,
    "Slytherin": green,
    "Hufflepuff": yellow,
    "Gryffindor": red,
}
