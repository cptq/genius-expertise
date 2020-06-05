from datetime import datetime

DATAPATH = "data"
FIGPATH = "figures"

# user stats
STATS = ("Annotations", "Suggestions", "Transcriptions", "Questions",
             "Answers", "Pyongs")

# user stats including iq
ALL_STATS = ("iq", "Annotations", "Suggestions", "Transcriptions", "Questions",
             "Answers", "Pyongs")

# user stats which we plot for publication
PUB_STATS = ("iq", "Annotations")
             
# roles that users may assume
ROLES = ("Contributor", "Editor", "Mediator", "Moderator", "Staff")
# roles including verified artists
A_ROLES = ("Verified Artist", "Contributor", "Editor", "Mediator", "Moderator", "Staff")

GENIUS_LAUNCH_TIME = datetime(2009, 10, 20).timestamp()

# colors we use in plots
PRED = '#B33274'
CYCLE_COLORS = ['#442AB8', '#B80717', '#149934']
