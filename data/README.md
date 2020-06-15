## Description
This data was obtained from [genius.com](https://genius.com) between September 2019 to January 2020. For more information about the site and its content, we recommend checking the [about page](https://genius.com/Genius-about-genius-annotated), [How Genius works page](https://genius.com/Genius-how-genius-works-annotated), and the [FAQ](https://genius.com/Genius-users-genius-faq-annotated). Most of the files are in the JSON lines format, meaning that each line break separates an entry in JSON format.

## Users

`user_info.jl` contains information about 424,474 users on the site. Each user has a `url_name` so that a user with name "streetlights" has a user page at https://genius.com/streetlights. This file contains each user's IQ, the roles that they have on the site (e.g. Editor, Contributor), and the number of annotations, suggestions, transcriptions, quesetions, answers, and pyongs that they have contributed to the site. `follows.csv` contains following relationships between users; for each row, the user in the first column follows the user in the second column. In total, we collected a social network of 784,432 users with 1,777,351 directed following relationships.

*Note*: the file map\_artist\_names.csv is used to make artist naming more consistent. Sometimes, Genius uses names for artists in certain url links on the site that are different from the url of the artist page after redirecting to it. When there are disagreements, the first column of the csv contains instances of an artist name that is used in some link on the site, and the second column contains the name of the artist on that artist's page. This mapping of names is done automatically in `load_data.py`.

## Annotations

`annotation_info.jl` contains annotations on lyrics. Each annotation has a timestamp, metadata on whethere it has been reviewed or even verified by an artist, the number of votes, number of pyongs, its url, the song it is associated to, the lyrics it is associated to, and the entire edit history. The content of the annotation is saved as the raw HTML displayed on the site. This content is saved over all of the annotations iterations (i.e. over all edits of the annotation). In total, there are 393,954 annotations in the dataset; of these, 322,613 are reviewed and there are 869,763 edits associated to these reviewed annotations. We also include the transcribed lyrics as they appear on Genius for songs associated to these annotations in `lyrics.jl`.

## Artists and Songs

We collect some other data on artist and song pages on Genius. In `artist_info.jl`, for 142,913 artists, we have: the "url name", number of followers on their Genius artist page, and a list of songs that they contributed to. In `song_info.jl`, for 223,257 songs, we have: the "url name", primary artist, views on the page, number of contributing users to the song, and genre tags.



