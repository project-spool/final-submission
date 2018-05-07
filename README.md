# Project Spool
### Tyler Angert, Angela He, Rahul Nair

## Project structure

```
|--data
|----cleaned
|----pickles
|--src
|----results
|----cluster.py
|----frequent_items.py
|----globals.py
|----process.py
|----pipeline.py
|--viz
|----data
|----index.html
```

## Downloading the data
Some of the data files are pretty large for this, so we uploaded the relevant ones to an Amazon S3 bucket for you to download. The program won't work unless you place them into the proper directories.

### Links
https://s3.amazonaws.com/project-spool-data/user-artist-df-pickle.pkl
https://s3.amazonaws.com/project-spool-data/user-id-groups.pkl

So, drag both of these `.pkl` files into the `data/pickles` folder, and you should be good to go. The other data sets are already included in the `data` folder.

For reference, here is the original data set:
http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html

## Clustering Compilation
Once you `cd` into the `src` folder, simply run `python3 pipeline.py` and make sure your computer has python 3.6.4 installed (this was developed using the Anaconda python distribution).

In order to adjust the clustering parameters, simply go into the `test` method inside of `pipeline.py` and play with any of the parameter variables as labeled.

Once the program is finished running, you can find all of the results in `json` format inside `src/results`, with each filed labeled appropriately by what data it represents.

## Visualization Compilation
Inside of `viz` contains an `index.html` file which contains a complete interactive data visualization of our best clustering results. To most easily compile this, open the `viz` folder with the text editor Brackets and create a live preview.

The visualization requires a local server in order to dynamically read in `json` files, so you can also run a simple server by running something like `python -m SimpleHTTPServer 8888 &` and then loading the page on `localhost:8888`, but I haven't thoroughly tested this. So opening the viz inside of Brackets is probably the best move.

A link to download Brackets if you don't have it: http://brackets.io. The live preview is the top right lightning bolt button.

## Notes
Currently, the program works by reading from a large pickled pandas data frame made from the original ~1.7GB data set. It is referenced to speed up the program. A pickle for all of the American users have also been pickled, as well as the user-id groups from the original dataset. Inside of the `data/cleaned` folder, you can see the filtered American TSV user file used to create the pickle.
