"""
TODO: 
Automate method for varying alpha, for 
S = (S*overlap)/(overlap + alpha)

FIGURE OUT RESIDUALS 

loading model probe and quiz ratings.

folder creation, logfile, model binding, parameter searching (for alpha & numNN)

See if it works at all?
"""
cimport cython
cimport numpy as np
import numpy as np

ctypedef np.float32_t f32_t
ctypedef np.int8_t i8_t
ctypedef np.int16_t i16_t
ctypedef np.int32_t i32_t
ctypedef np.int64_t i64_t

cdef inline float cmin(float a, float b): return a if a <= b else b


class KNN(object):
    """
    Wrapper class for a lot of kNN functionality

    Includes ability to create similarity matrices,
    and also to make predictions from them using
    either raw ratings or residuals from other models.
    """
    def __init__(self, mode, knndir='/home/playmice/ML/Projects/Netflix/netflix_prize/results/knn/'):
        super(KNN, self).__init__()
        self.knndir = knndir

        self.parameters = ['numNN', 'alpha', 'useresiduals', 'simName']

        # Set some intelligent defaults...
        self.numNN = 30
        self.alpha = 1
        self.useresiduals = False
    # self.simType = 'movie' # this doesn't matter for now...

    # ...but you'll still need to load
    # self.S
    # self.fd
    # at least one of [self.qd, self.td]

    def knn(self, haveknn=False):
        """
        Make predictions using this fully-initialized KNN class.
        Will spit out appropriate errors if you're missing
        certain variables we've gotta know to make a prediction.

        TODO: make this. lololol.
        """
        # Make sure we've got everything loaded
        # NOTE: The check that self.fd.userResiduals exists happens in the function that uses it.
        requirements = ['numNN', 'alpha', 'useresiduals', 'fd', 'S', 'simName']
        for r in requirements:
            assert hasattr(self, r)

        if not hasattr(self, 'qd') and not hasattr(self, 'td'):
            raise ValueError, 'Must have either qd or td. Neither found'

        if self.useresiduals is True:
            if hasattr(self, 'qd')
                assert hasattr(self, 'modelQuizRatings')
            if hasattr(self, 'td'):
                assert hasattr(self, 'modelProbeRatings')

        # NOTE: we're only doing movie-based similarity predictions.
        # When I write any user-based similarity functions, I'll add a class variable as
        # self.simType = 'user' or self.simType = 'movie'
        # that'll allow easy switching between the both types.



        # OK GO

        self.makeKNNLog()

        # Find the K-nearest neighbors
        if haveknn is False:
            print('Finding k-nearest neighbors with similarity measure %s...' % self.simName)
            self.knn = findNearestNeighbors(S, np.arange(17770, dtype='int32'), self.numNN)
        if haveknn is True:
            assert hasattr(self, 'knn')

        if hasattr(self, 'td'):
            print('\nPredicting ratings for probe data...\n')
            probe10ratings = self.predictRatingsForMovieSims(self.td)
        elif hasattr(self, 'qd'):
            print('\nPredicting ratings for quiz data...\n')
            quizratings = self.predictRatingsForMovieSims(self.qd)

        if self.useresiduals is True:
            if hasattr(self, 'td'):
                probe10ratings += self.modelProbeRatings
            elif hasattr(self, 'qd'):
                quizratings += self.modelQuizRatings

    def predictRatingsForMovieSims(self, qd):
        """
        Predict ratings using a movie-movie based similarity.

        VARIABLES:
        qd - FlixData class for the quiz/probe data

        USAGE:
        predictions = KNN.predictRatings(qd)

        Returns an array of size qd.numratings.
        Uses knn.S, knn.numNN, and knn.useresiduals to make the prediction.

        knn.useresiduals should be either True or False.
        """
        fd = self.fd
        cdef np.ndarray[f32_t, ndim=2] S = self.S
        cdef float topsum, bottomsum
        cdef int iuser, imovie
        cdef int numNN = self.numNN
        cdef int numitems = S.shape[0]
        cdef int numratings = qd.numratings
        cdef np.ndarray[i32_t, ndim=1] userIDs = qd.userIDsForUsers
        cdef np.ndarray[i16_t, ndim=1] movieIDs = qd.movieIDs
        cdef np.ndarray[f32_t, ndim=1] quizratings = np.zeros(numratings, dtype='float32')

        # Check if we're working on residuals or not...
        useresiduals = self.useresiduals
        if useresiduals is True:
            getRating = self.getRatingForUserAndMovie
            assert hasattr(fd, 'userResiduals')  # make sure there are residuals at all.
            assert hasattr(qd, 'userResiduals')
        elif useresiduals is False:
            getRating = self.getResidualForUserAndMovie
        else:
            raise ValueError, '"useresiduals" variable can only be True or False.'

        print('Finding the nearest neighbors...')
        cdef np.ndarray[i32_t, ndim=2] knn = self.knn

        print('Predicting ratings on the quiz set...')
        for i from 0 <= i < numratings:
            iuser = userIDs[i]
            imovie = movieIDs[i]
            topsum = 0.0
            bottomsum = 0.0
            for j from 0 <= j < numNN:
                topsum += S[imovie, knn[imovie, j]] * getRating(iuser, knn[imovie, j], fd)

            for j from 0 <= j < numNN:  # separate loops in the hope of vectorization
                bottomsum += S[imovie, knn[imovie, j]]

            quizratings[i] = topsum / bottomsum

        return quizratings

    def loadSims(self, simName):
        """
        Load in a similarity matrix found in KNN.knndir
        USAGE:
        KNN.loadSims('normeuclidean')

        KNN.S holds the similarity matrix.
        KNN.simType holds the name of the metric
            used to generate the similarity matrix.
        """

        from os.path import join
        self.S = np.load(join(self.knndir, simName))
        self.simName = simName

    def loadSharedSupport(self):
        """
        Load the precomputed sharedsupport of each movie.
        """
        from os.path import join
        self.sharedsupport = np.load(join(self.knndir, 'sharedsupport'))

    def makeSimsFromSVD(self, knnmodelname, svdmodelname, simtype='normeuclidean',
                        resultdir='/home/playmice/ML/Projects/Netflix/netflix_prize/results/'):
        """
        Using the movie-matrix from an SVD model, creating a similarity matrix.
        The similarity matrix can be used for kNN predictions later.

        USAGE:
        knnmodelname = 'normeuclidKNN_1000featureBRISMF'
        svdmodelname = '1000featureBRISMF'
        KNN.makeSimsFromSVD(knnmodelname, svdmodelname, simtype='normeuclidean', resultdir='../results/')
        """
        from svd import loadModel
        from os import makedirs
        from os.path import join

        self.svdmodelname = svdmodelname
        self.knnmodelname = knnmodelname
        U, M = loadModel(svdmodelname, resultdir)
        del U

        if simtype == 'normeuclidean':
            self.S = makeNormalizedEuclideanSimMatrix(M)
        elif simtype == 'normscalar'
            self.S = makeNormalizedScalarSimMatrix(M)
        elif simtype = 'euclidean':
            self.S = makeEuclideanSimMatrix(M)
        else:
            raise TypeError, 'Unrecognized metric type %s' % simtype
        del M

        # Make a home for the sims
        outputdir = join(resultdir, 'knn', knnmodelname)
        makedirs(outputdir)

        # Put the sims in their home
        self.S.dump(join(outputdir, knnmodelname))

        # Use those sims to make predictions on raw ratings
        # DOIT
        self.S *= overlap / (overlap + alpha)

        # Make a logfile with a couple details for posterity
        logfile = open(join(outputdir, 'log.txt'), 'wt')
        logfile.write('KNN Similarity Matrix\n')
        logfile.write('Derived from SVD model: %s\n' % self.svdmodelname)
        logfile.write('Similarity metric: %s\n' % simtype)
        logfile.close()

    predictRatings(fd, qd, np.ndarray[f32_t, ndim = 2] S, np.ndarray[i16_t, ndim = 2] overlap, float
    alpha = 0, int
    numNN = 30):

    def getRatingForUserAndMovie(int userid, int movieid, fd):
        cdef np.ndarray[i32_t, ndim=2] userIndex = fd.userIndex
        cdef np.ndarray[i16_t, ndim=1] movieIDs = fd.movieIDs
        cdef np.ndarray[i8_t, ndim=1] userRatings = fd.userRatings

        cdef int start, end, i, done
        i = 0
        done = 0

        start = userIndex[userid, 0]
        maxi = userIndex[userid, 1] - start

        while done == 0:
            if movieIDs[start + i] == movieid:
                return userRatings[start + i]
            elif i == maxi:
                return None
            else:
                i += 1

    def getResidualForUserAndMovie(int userid, int movieid, fd):
        cdef np.ndarray[i32_t, ndim=2] userIndex = fd.userIndex
        cdef np.ndarray[i16_t, ndim=1] movieIDs = fd.movieIDs
        cdef np.ndarray[i8_t, ndim=1] userResiduals = fd.userResiduals

        cdef int start, end, i, done
        i = 0
        done = 0

        start = userIndex[userid, 0]
        maxi = userIndex[userid, 1] - start

        while done == 0:
            if movieIDs[start + i] == movieid:
                return userResiduals[start + i]
            elif i == maxi:
                return None
            else:
                i += 1

    def getRatingForMovieAndUser(int userid, int movieid, fd):
        cdef np.ndarray[i32_t, ndim=2] movieIndex = fd.movieIndex
        cdef np.ndarray[i16_t, ndim=1] userIDs = fd.userIDs
        cdef np.ndarray[i8_t, ndim=1] movieRatings = fd.movieRatings

        cdef int start, end, i, done
        i = 0
        done = 0

        start = movieIndex[userid, 0]
        maxi = movieIndex[userid, 1] - start

        while done == 0:
            if userIDs[start + i] == userid:
                return movieRatings[start + i]
            elif i == maxi:
                return None
            else:
                i += 1

    # ====================================================
    ## Distance metric calculation functions
    # ====================================================

    def makeNormalizedEuclideanSimMatrix(np.ndarray[f32_t, ndim=2] featureMatrix):
        cdef int i, k, kk
        cdef float tmpsum
        cdef int numitems = np.shape(featureMatrix)[0]
        cdef int numfeatures = np.shape(featureMatrix)[1]

        cdef np.ndarray[f32_t, ndim=2] S = np.zeros((numitems, numitems), dtype='float32')  # the similarity matrix

        # Calculate the root of the squared-sum magnitudes.
        cdef np.ndarray[f32_t, ndim=1] sqsum = np.zeros(numitems, dtype='float32')
        for i from 0 <= i < numitems:
            for k from 0 <= k < numfeatures:
                sqsum[i] += featureMatrix[i, k] ** 2
        sqsum = np.sqrt(sqsum)

        # Find the similarities
        for i from 0 <= i < numitems - 1:
            if i % 1000 == 0: print('Working on item %d' % i)
            for k from i + 1 <= k < numitems:
                tmpsum = 0.0
                for kk from 0 <= kk < numfeatures:
                    tmpsum += (featureMatrix[i, kk] - featureMatrix[k, kk]) ** 2
                S[i, k] = tmpsum / (sqsum[i] * sqsum[k])
                S[k, i] = S[i, k]  # filling in the lower triangle will make later operations easier.
        return S

    def makeNormalizedScalarSimMatrix(np.ndarray[f32_t, ndim=2] featureMatrix):
        cdef int i, k, kk
        cdef float tmpsum
        cdef int numitems = np.shape(featureMatrix)[0]
        cdef int numfeatures = np.shape(featureMatrix)[1]

        cdef np.ndarray[f32_t, ndim=2] S = np.zeros((numitems, numitems), dtype='float32')  # the similarity matrix

        # Calculate the root of the squared-sum magnitudes.
        cdef np.ndarray[f32_t, ndim=1] sqsum = np.zeros(numitems, dtype='float32')
        for i from 0 <= i < numitems:
            for k from 0 <= k < numfeatures:
                sqsum[i] += featureMatrix[i, k] ** 2
        sqsum = np.sqrt(sqsum)

        # Find the similarities
        for i from 0 <= i < numitems - 1:
            if i % 1000 == 0: print('Working on item %d' % i)
            for k from i + 1 <= k < numitems:
                tmpsum = 0.0
                for kk from 0 <= kk < numfeatures:
                    tmpsum += featureMatrix[i, kk] * featureMatrix[k, kk]
                S[i, k] = tmpsum / (sqsum[i] * sqsum[k])
                S[k, i] = S[i, k]  # filling in the lower triangle will make later operations easier.
        return S

    def makeEuclideanSimMatrix(np.ndarray[f32_t, ndim=2] featureMatrix):
        cdef int i, k, kk
        cdef float tmpsum
        cdef int numitems = np.shape(featureMatrix)[0]
        cdef int numfeatures = np.shape(featureMatrix)[1]

        cdef np.ndarray[f32_t, ndim=2] S = np.zeros((numitems, numitems), dtype='float32')  # the similarity matrix

        # Find the similarities
        for i from 0 <= i < (numitems - 1):
            if i % 1000 == 0: print('Working on item %d' % i)
            for k from i + 1 <= k < numitems:
                tmpsum = 0.0
                for kk from 0 <= kk < numfeatures:
                    tmpsum += (featureMatrix[i, kk] - featureMatrix[k, kk]) ** 2
                S[i, k] = tmpsum
                S[k, i] = S[i, k]  # filling in the lower triangle will make later operations easier.
        return S

    # ====================================================
    ## Miscellaneous / Experimental Functions
    # ====================================================

    def makeSupportSimMatrix(fd):
        """
        sim_ij = intersect(N(i), N(j)) / min(|N(i)|, |N(j)|)
        """
        cdef np.ndarray[i32_t, ndim=2] movieindex = fd.movieIndex
        cdef np.ndarray[i32_t, ndim=1] userids = fd.userIDs
        cdef float count, numratedi, numratedk
        cdef int i, k, ii, kk
        cdef int numitems = fd.movieIndex.shape[0]

        cdef np.ndarray[f32_t, ndim=2] S = np.zeros((numitems, numitems), dtype='float32')
        cdef np.ndarray[i32_t, ndim=2] overlap = np.zeros((numitems, numitems), dtype='int32')

        for i from 0 <= i < numitems - 1:
            if i % 10 == 0: print('Working on item %d' % i)
            numratedi = movieindex[i, 1] - movieindex[i, 0]
            for k from i + 1 <= k < numitems:
                numratedk = movieindex[k, 1] - movieindex[k, 0]
                overlap[i, k] = np.intersect1d(userids[fd.ui(i)], userids[fd.ui(k)]).size
                overlap[k, i] = overlap[i, k]

                S[i, k] = overlap[i, k] / cmin(numratedi, numratedk)
                S[k, i] = S[i, k]
        return S, overlap

    def makeUserRatingHash(fd):
        cdef np.ndarray[i32_t, ndim=2] movieIndex = fd.movieIndex
        cdef np.ndarray[i32_t, ndim=1] userIDs = fd.userIDs
        userRatingHash = []
        cdef int i
        for i from 0 <= i < 17770:
            if i % 1000 == 0: print('Analyzing item %d' % i)
            idict = dict(np.vstack((fd.userIDs[fd.mi(i)], fd.movieRatings[fd.mi(i)])).T)
            userRatingHash.append(idict)
        return userRatingHash

    def findCommonUsers(fd):
        """
        Make a list commonUsers = []
        For each item 0 <= i < 17770, item i < k < 17770
        Turn the keys into a set.
        Find the intersection, save it in commonUsers[i]
        """
        commonUsers = []
        userSets = []  # indexed by movie, ya
        cdef int i, k

        print('Finding common users...')
        for i from 0 <= i < 17770:
            print('\tAnalyzing user %d' % i)
            commonUsers.append([])
            usersi = fd.userIDs[fd.mi(i)]
            for k from i < k < 17770:
                commonUsers[i].append(np.intersect1d(usersi, fd.userIDs[fd.mi(k)]))

        return commonUsers

    def Pearson2(fd, nummovies = 17770):
        """
        This function is so hacky. It uses a lot of builtin Numpy functions.
        Ideally, I should be handwriting the indexing, but it's more difficult
        than I really know how to do at this point. The penalty I pay is in time.
        SUPER SLOW.
        AND
        UNREGULARIZED.
        UGH.


        Make a list commonUsers = []
        For each item 0 <= i < 17770, item i < k < 17770
        Turn the keys into a set.
        Find the intersection, save it in commonUsers[i]
        """
        cdef int i, k
        cdef int numitems = nummovies
        cdef np.ndarray[i32_t, ndim=1] usersi
        cdef np.ndarray[i32_t, ndim=1] usersk
        cdef np.ndarray[i64_t, ndim=1] rangei
        cdef np.ndarray[i64_t, ndim=1] rangek
        cdef np.ndarray[i64_t, ndim=1] sortorderi
        cdef np.ndarray[i64_t, ndim=1] sortorderk
        cdef np.ndarray[f32_t, ndim=1] ratingsi
        cdef np.ndarray[f32_t, ndim=1] ratingsk
        cdef np.ndarray[f32_t, ndim=2] S = np.zeros((numitems, 17770), dtype='float32')
        cdef np.ndarray[i8_t, ndim=1] movieRatings = fd.movieRatings
        cdef np.ndarray[i32_t, ndim=2] movieIndex = fd.movieIndex
        cdef np.ndarray[f32_t, ndim=1] meanRating = np.zeros(17770, dtype='float32')
        cdef np.ndarray[f32_t, ndim=1] stdDev = np.zeros(17770, dtype='float32')

        print('Calculating movie averages and standard deviations...')
        for i from 0 <= i < numitems:
            meanRating[i] = np.mean(movieRatings[movieIndex[i, 0]:movieIndex[i, 1]]).astype('float32')
            stdDev[i] = np.std(movieRatings[movieIndex[i, 0]:movieIndex[i, 1]]).astype('float32')

        print('Finding common users...')
        for i from 0 <= i < numitems:
            print('Analyzing user %d' % i)
            rangei = fd.mi(i)
            sortorderi = fd.userIDs[rangei].argsort()
            usersi = fd.userIDs[rangei][sortorderi]
            for k from i < k < 17770:
                rangek = fd.mi(k)
                sortorderk = fd.userIDs[rangek].argsort()
                usersk = fd.userIDs[rangek][sortorderk]

                ratingsi = movieRatings[rangei][sortorderi][np.setmember1d(usersi, usersk)].astype('float32')
                ratingsk = movieRatings[rangek][sortorderk][np.setmember1d(usersk, usersi)].astype('float32')
                S[i, k] = np.sum(
                    (ratingsi - meanRating[i]) * (ratingsk - meanRating[k]) / (stdDev[i] * stdDev[k])) / ratingsi.size

        return S

    @cython.boundscheck(False)
    def makePearsonCorrSimMatrix(fd):
        """
        I clearly don't have this figured out.
        Pearson correlations is on hold until I learn a bit more about
        proper lookup and sorting algorithms. This is programming
        more serious than I'm used to.
        The slowest part is finding the overlapping users and their ratings

        TODO:
        Preslice userID and rating arrays.
        Stick everything into a big list
        Use unsigned ints for indexing.



        From On the Gravity Recommendation System, Gabor Takacs et al. 2007
        """
        # for all common users between movies i and j
        # score = sum of
        # (userRating[i] - meanRating[i])(userRating[j] - meanRating[j]) / (stdDev[i]*stdDev[j])
        cdef unsigned int i, k, ii, kk, useri, userk
        cdef int numratedOne, numratedTwo
        cdef float tmpsum, ratingi, ratingk, numCommonRaters
        cdef int numitems = 17770  # hard-coded for now...
        cdef np.ndarray[i8_t, ndim=1] movieRatings = fd.movieRatings
        cdef np.ndarray[i32_t, ndim=1] userIDs = fd.userIDs
        cdef np.ndarray[f32_t, ndim=1] meanRating = np.zeros(numitems, dtype='float32')
        cdef np.ndarray[f32_t, ndim=1] stdDev = np.zeros(numitems, dtype='float32')
        cdef np.ndarray[i32_t, ndim=1] ratersOne
        cdef np.ndarray[i32_t, ndim=1] ratersTwo
        cdef np.ndarray[i32_t, ndim=2] movieIndex = fd.movieIndex
        cdef np.ndarray[f32_t, ndim=2] S = np.zeros((numitems, numitems), dtype='float32')

        # Calculate movie means and stds
        print('Calculating movie averages and standard deviations...')
        for i from 0 <= i < numitems:
            meanRating[i] = np.mean(movieRatings[movieIndex[i, 0]:movieIndex[i, 1]]).astype('float32')
            stdDev[i] = np.std(fd.movieRatings[movieIndex[i, 0]:movieIndex[i, 1]]).astype('float32')

        print('Doing pairwise Pearson correlations...')
        print('This will take awhile...')

        for i from 0 <= i < numitems - 1:
            print('Working on item %d' % i)
            numratedOne = movieIndex[i, 1] - movieIndex[i, 0]
            ratersOne = userIDs[movieIndex[i, 0]:movieIndex[i, 1]]

            for k from i + 1 <= k < numitems:
                numratedTwo = movieIndex[k, 1] - movieIndex[k, 0]
                ratersTwo = userIDs[movieIndex[k, 0]:movieIndex[k, 1]]

                numCommonRaters = 0.0
                tmpsum = 0.0

                for ii from 0 <= ii < numratedOne:
                    useri = <unsigned int> ratersOne[ii]
                    if useri in ratersTwo:  # is this fast enough? who cares, just code...
                        numCommonRaters += 1
                        ratingi = movieRatings[movieIndex[i, 0] + ii]

                        for kk from 0 <= kk < numratedTwo:  # scan for the common ID (this is a h4xx0r)
                            if userIDs[movieIndex[k, 0] + kk] == useri:
                                ratingk = movieRatings[movieIndex[k, 0] + kk]
                                tmpsum += (ratingi - meanRating[i]) * (ratingk - meanRating[k]) / (
                                stdDev[i] * stdDev[k])

                S[i, k] = tmpsum / numCommonRaters
                S[k, i] = S[i, k]

                S.dump('PearsonSimMatrix')  # just so that we can get intermediate results

        return S  # just for testing

    def findNearestNeighbors(np.ndarray[f32_t, ndim=2] S, np.ndarray[i32_t, ndim=1] itemlist, int k):
        """
        Find the k nearest neighbors for every item in itemlist.
        """
        cdef int i
        cdef int numNNitems = np.size(itemlist)
        cdef int numitems = np.shape(S)[0]
        cdef np.ndarray[f32_t, ndim=2] nn_dist = np.zeros((numNNitems, k), dtype='float32')  # the distance list
        cdef np.ndarray[i32_t, ndim=2] nn = np.zeros((numNNitems, k), dtype='int32')  # the nearest-neighbors list

        # STRATEGY:
        # make sure any 0 (self-similar) values are set to N.Inf
        # apply NumPy's argmin() function to find minimum indices.
        # pull out the actual values from S and save them.
        # set the lowest values to N.Inf, and continue to the next-nearest value.
        # After all that's done, replace the changed values to their original values.

        for i from 0 <= i < numitems:
            S[i, i] = np.Inf

        for i from 0 <= i < k:
            if i % 1000 == 0 & i > 0: print('Analyzing item %d' % i)
            nn[:, i] = np.argmin(S[itemlist, :], 1)
            nn_dist[:, i] = S[itemlist, nn[:, i]]
            S[itemlist, nn[:, i]] = np.Inf

        for i from 0 <= i < k:
            S[itemlist, nn[:, i]] = nn_dist[:, i]

        for i from 0 <= i < numitems:
            S[i, i] = 0.0

        return nn
