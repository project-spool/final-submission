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
|----utils.py
|----pipeline.py
|--viz
|----data
|----index.html
```

## Compilation
Once you `cd` into the `src` folder, simply run `python3 pipeline.py` and make sure your computer has python 3.6.4 installed (this was developed using the Anaconda python distribution).

In order to adjust the clustering parameters, simply go into the `test` method inside of `pipeline.py` and play with any of the variables. 

Once the program is finished running, you can find all of the results in `json` format inside `src/results`, with each filed labeled appropriately by what data it represents.

## Notes
Currently, the program works by reading from a large pickled pandas data frame made from the original ~1.7GB data set. It is referenced to speed up the program. A pickle for all of the American users have also been pickled, as well as the user-id groups from the original dataset. Inside of the `data/cleaned` folder, you can see the filtered American TSV user file used to create the pickle.