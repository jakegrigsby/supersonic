import os
os.chdir('..')

from supersonic import paramsearch

if __name__ == "__main__":
    searcher = paramsearch.AgentParamFinder(50)
    searcher.find_params()