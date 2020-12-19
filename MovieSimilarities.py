# To run locally:
# !python MovieSimilarities.py --files=ml-100k/u.item ml-100k/u.data > sims.txt

# To run on a single EMR node:
# !python MovieSimilarities.py -r emr --files=ml-100k/u.item ml-100k/u.data

# To run on 4 EMR nodes:
#!python MovieSimilarities.py -r emr --num-ec2-instances=4 --files=ml-100k/u.item ml-100k/u.data

# Troubleshooting EMR jobs (subsitute your job ID):
# !python -m mrjob.tools.emr.fetch_logs --find-failure j-1NXMMBNEQHAFT

# u.data: userID movieID rating timestamp
# u.item: movieId | movieNames

from mrjob.job import MRJob
from mrjob.step import MRStep
from math import sqrt
from itertools import combinations
import time



class MovieSimilarities(MRJob):
    def configure_args(self):
        super(MovieSimilarities, self).configure_args()
        # Add a command-line option that sends an external file to Hadoop
        self.add_file_arg('--db', help='Path to u.item')

    # Output {movieId:movieName}
    def load_movie_names(self):
        # Load database of movie names.
        self.movieNames = {}

        with open("u.item", encoding='ascii', errors='ignore') as f:
            for line in f:
                fields = line.split('|')
                self.movieNames[int(fields[0])] = fields[1]
    
    # Make a multi-step job
    def steps(self):
        return [
            MRStep(mapper=self.mapper_parse_input,
                    reducer=self.reducer_ratings_by_user),
            MRStep(mapper=self.mapper_create_item_pairs,
                    reducer=self.reducer_compute_similarity),
            MRStep(mapper=self.mapper_sort_similarities,
                   # Define an action to run before the mapper processes any input. 
                   mapper_init=self.load_movie_names,
                    reducer=self.reducer_output_similarities)]
    
    '''STEP 1'''
    # Input u.data line
    # Output userID => (movieID, rating)
    def mapper_parse_input(self, key, line): # By default key will be None        
        (userID, movieID, rating, timestamp) = line.split('\t')
        yield  userID, (movieID, float(rating))
        
    
    # Input userID => (movieID, rating)
    # Output grouped userId => ratings[(movieID, rating)]
    def reducer_ratings_by_user(self, userID, IDrating):
        global mapper1end
        mapper1end = time.process_time()
        
        #Group (item, rating) pairs by userID
        IDratings = []
        for movieID, rating in IDrating:
            IDratings.append((movieID, rating))
        yield userID, IDratings
    
    
    '''STEP 2'''
    # Input userId => IDratings[(movieID, rating)]
    # Output (movieID1, movieID2) => (rating1, rating2)
    def mapper_create_item_pairs(self, userID, IDratings):
        # Find every pair of movies each user has seen, and emit
        # each pair with its associated ratings

        # "combinations" finds every possible pair from the list of movies
        # this user viewed.
        global reducer1end
        reducer1end = time.process_time()
        
        for IDrating1, IDrating2 in combinations(IDratings, 2):
            movieID1 = IDrating1[0]
            rating1 = IDrating1[1]
            movieID2 = IDrating2[0]
            rating2 = IDrating2[1]

            # Produce both orders so sims are bi-directional
            yield (movieID1, movieID2), (rating1, rating2)
            yield (movieID2, movieID1), (rating2, rating1)

    # Computes the cosine similarity metric between two rating vectors.
    def cosine_similarity(self, ratingPairs):       
        numPairs = 0
        sum_xx = sum_yy = sum_xy = 0
        for ratingX, ratingY in ratingPairs:
            sum_xx += ratingX * ratingX
            sum_yy += ratingY * ratingY
            sum_xy += ratingX * ratingY
            numPairs += 1

        numerator = sum_xy
        denominator = sqrt(sum_xx) * sqrt(sum_yy)

        score = 0
        if (denominator):
            score = (numerator / (float(denominator)))

        return (score, numPairs)
    
    # Input (movieID1, movieID2) => (rating1, rating2)
    # Output grouped(movieID1, movieID2) => (score, numPairs)
    def reducer_compute_similarity(self, moviePair, ratingPairs):
        # Compute the similarity score between the ratings vectors
        # for each movie pair viewed by multiple people
        global mapper2end
        mapper2end = time.process_time()
        
        score, numPairs = self.cosine_similarity(ratingPairs)

        # Set minimum score and number of co-ratings to ensure quality
        if (numPairs > 70 and score > 0.98):
            yield moviePair, (score, numPairs)
    
    
    '''STEP 3'''
    # Input (movieID1, movieID2) => (score, numPairs)
    # Output (movieName1, score) => (movieName2, numPairs)
    def mapper_sort_similarities(self, moviePair, scores):   
        # Shuffle things around so the key is (movie1, score)
        # so we have meaningfully sorted results.
        global reducer2end
        reducer2end = time.process_time()
        
        score, numPairs = scores
        movie1, movie2 = moviePair

        yield (self.movieNames[int(movie1)], score), \
            (self.movieNames[int(movie2)], numPairs)
    
    # Input (movieName1, score) => (movieName2, numPairs)
    # Output movieName1 => (movieName2, score, numPairs)
    def reducer_output_similarities(self, NameScore, NamePair):
        global mapper3end
        mapper3end = time.process_time()
        
        movie1, score = NameScore
        for movie2, numPairs in NamePair:
            yield movie1, (movie2, score, numPairs)


if __name__ == '__main__':
    #start
    start = time.process_time()
    mapper1end = 0
    reducer1end = 0
    mapper2end = 0
    reducer2end = 0
    mapper3end = 0
    MovieSimilarities.run()
    end = time.process_time()
    print('till mapper1end:', str(mapper1end - start))
    print('till reducer1end:', str(reducer1end - mapper1end))
    print('till mapper2end:', str(mapper2end - reducer1end))
    print('till reducer2end:', str(reducer2end - mapper2end))
    print('till mapper3end:', str(mapper3end - reducer2end))
    print('till reducer3end:', str(end - mapper3end))
    print('total:', str(end-start))
