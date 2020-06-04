from datetime import datetime

datapath = "spiders/data"
figpath = "figures"

STATS = ("Annotations", "Suggestions", "Transcriptions", "Questions",
             "Answers", "Pyongs")
# user stats
ALL_STATS = ("iq", "Annotations", "Suggestions", "Transcriptions", "Questions",
             "Answers", "Pyongs")
# user stats which we plot for publication
PUB_STATS = ("iq", "Annotations")

             
ROLES = ("Contributor", "Editor", "Mediator", "Moderator", "Staff")
A_ROLES = ("Verified Artist", "Contributor", "Editor", "Mediator", "Moderator", "Staff")

# roles which we plot for publication
PUB_ROLES = ('Editor', 'Mediator', 'Moderator', 'Staff')

GENIUS_LAUNCH_TIME = datetime(2009, 10, 20).timestamp()

# colors
PRED = '#B33274'
CYCLE_COLORS = ['#442AB8', '#B80717', '#149934']