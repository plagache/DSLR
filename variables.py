# https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)

# HYPERPARAMETERS
number_of_fold = 6
sampling = 0
stochastic = False
steps = 2000
# learning_rate = 4
learning_rate = 0.4
scheduler = None
scheduler_type = ["exp", "linear", None]
learning_rate_decay = 0.001


# FEATURES
labels_column = "Hogwarts House"
# 'Ancient Runes',
# 'Arithmancy',
# 'Astronomy',
# 'Best Hand',
# 'Birthday',
# 'Care of Magical Creatures',
# 'Charms',
# 'Defense Against the Dark Arts',
# 'Divination',
# 'First Name',
# 'Flying',
# 'Herbology',
# 'History of Magic',
# 'Hogwarts House',
# 'Index',
# 'Last Name',
# 'Muggle Studies',
# 'Potions',
# 'Transfiguration'

# alex
    # "Ancient Runes",
    # "Astronomy",
    # "Charms",
    # "Defense Against the Dark Arts",
    # "Divination",
    # "Herbology",
# visual intuition
    # "Ancient Runes",
    # "Astronomy",
    # "Charms",
    # "Defense Against the Dark Arts",
    # "Divination",
    # "Flying",
    # "Herbology",

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
