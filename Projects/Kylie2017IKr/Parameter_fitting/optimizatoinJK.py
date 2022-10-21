import numpy as np
import pints


class OptimisationController(object):
    """
    Finds the parameter values that minimise an :class:`ErrorMeasure` or
    maximise a :class:`LogPDF`.

    Parameters
    ----------
    function
        An :class:`pints.ErrorMeasure` or a :class:`pints.LogPDF` that
        evaluates points in the parameter space.
    x0
        The starting point for searches in the parameter space. This value may
        be used directly (for example as the initial position of a particle in
        :class:`PSO`) or indirectly (for example as the center of a
        distribution in :class:`XNES`).
    sigma0
        An optional initial standard deviation around ``x0``. Can be specified
        either as a scalar value (one standard deviation for all coordinates)
        or as an array with one entry per dimension. Not all methods will use
        this information.
    boundaries
        An optional set of boundaries on the parameter space.
    transformation
        An optional :class:`pints.Transformation` to allow the optimiser to
        search in a transformed parameter space. If used, points shown or
        returned to the user will first be detransformed back to the original
        space.
    method
        The class of :class:`pints.Optimiser` to use for the optimisation.
        If no method is specified, :class:`CMAES` is used.
    """

    def __init__(
            self, function, x0, sigma0=None, boundaries=None,
            transformation=None, method=None):

        # Convert x0 to vector
        # This converts e.g. (1, 7) shapes to (7, ), giving users a bit more
        # freedom with the exact shape passed in. For example, to allow the
        # output of LogPrior.sample(1) to be passed in.
        x0 = pints.vector(x0)

        # Check dimension of x0 against function
        if function.n_parameters() != len(x0):
            raise ValueError(
                'Starting point must have same dimension as function to'
                ' optimise.')

        # Check if minimising or maximising
        self._minimising = not isinstance(function, pints.LogPDF)

        # Apply a transformation (if given). From this point onward the
        # optimiser will see only the transformed search space and will know
        # nothing about the model parameter space.
        if transformation is not None:
            # Convert error measure or log pdf
            if self._minimising:
                function = transformation.convert_error_measure(function)
            else:
                function = transformation.convert_log_pdf(function)

            # Convert initial position
            x0 = transformation.to_search(x0)

            # Convert sigma0, if provided
            if sigma0 is not None:
                sigma0 = transformation.convert_standard_deviation(sigma0, x0)
            if boundaries:
                boundaries = transformation.convert_boundaries(boundaries)

        # Store transformation for later detransformation: if using a
        # transformation, any parameters logged to the filesystem or printed to
        # screen should be detransformed first!
        self._transformation = transformation

        # Store function
        if self._minimising:
            self._function = function
        else:
            self._function = pints.ProbabilityBasedError(function)
        del(function)

        # Create optimiser
        if method is None:
            method = pints.CMAES
        elif not issubclass(method, pints.Optimiser):
            raise ValueError('Method must be subclass of pints.Optimiser.')
        self._optimiser = method(x0, sigma0, boundaries)

        # Check if sensitivities are required
        self._needs_sensitivities = self._optimiser.needs_sensitivities()

        # Logging
        self._log_to_screen = True
        self._log_filename = None
        self._log_csv = False
        self.set_log_interval()

        # Parallelisation
        self._parallel = False
        self._n_workers = 1
        self.set_parallel()

        # User callback
        self._callback = None

        # :meth:`run` can only be called once
        self._has_run = False

        #
        # Stopping criteria
        #

        # Maximum iterations
        self._max_iterations = None
        self.set_max_iterations()

        # Maximum unchanged iterations
        self._max_unchanged_iterations = None
        self._min_significant_change = 1
        self.set_max_unchanged_iterations()

        # Threshold value
        self._threshold = None

        # Post-run statistics
        self._evaluations = None
        self._iterations = None
        self._time = None
        
        self.update_record = []

    def evaluations(self):
        """
        Returns the number of evaluations performed during the last run, or
        ``None`` if the controller hasn't ran yet.
        """
        return self._evaluations


    def iterations(self):
        """
        Returns the number of iterations performed during the last run, or
        ``None`` if the controller hasn't ran yet.
        """
        return self._iterations


    def max_iterations(self):
        """
        Returns the maximum iterations if this stopping criterion is set, or
        ``None`` if it is not. See :meth:`set_max_iterations()`.
        """
        return self._max_iterations


    def max_unchanged_iterations(self):
        """
        Returns a tuple ``(iterations, threshold)`` specifying a maximum
        unchanged iterations stopping criterion, or ``(None, None)`` if no such
        criterion is set. See :meth:`set_max_unchanged_iterations()`.
        """
        if self._max_unchanged_iterations is None:
            return (None, None)
        return (self._max_unchanged_iterations, self._min_significant_change)


    def optimiser(self):
        """
        Returns the underlying optimiser object, allowing detailed
        configuration.
        """
        return self._optimiser


    def parallel(self):
        """
        Returns the number of parallel worker processes this routine will be
        run on, or ``False`` if parallelisation is disabled.
        """
        return self._n_workers if self._parallel else False


    def run(self):
        """
        Runs the optimisation, returns a tuple ``(xbest, fbest)``.

        An optional ``callback`` function can be passed in that will be called
        at the end of every iteration. The callback should take the arguments
        ``(iteration, optimiser)``, where ``iteration`` is the iteration count
        (an integer) and ``optimiser`` is the optimiser object.
        """
        # Can only run once for each controller instance
        if self._has_run:
            raise RuntimeError("Controller is valid for single use only")
        self._has_run = True

        # Check stopping criteria
        has_stopping_criterion = False
        has_stopping_criterion |= (self._max_iterations is not None)
        has_stopping_criterion |= (self._max_unchanged_iterations is not None)
        has_stopping_criterion |= (self._threshold is not None)
        if not has_stopping_criterion:
            raise ValueError('At least one stopping criterion must be set.')

        # Iterations and function evaluations
        iteration = 0
        evaluations = 0

        # Unchanged iterations count (used for stopping or just for
        # information)
        unchanged_iterations = 0

        # Choose method to evaluate
        f = self._function
        if self._needs_sensitivities:
            f = f.evaluateS1

        # Create evaluator object
        if self._parallel:
            # Get number of workers
            n_workers = self._n_workers

            # For population based optimisers, don't use more workers than
            # particles!
            if isinstance(self._optimiser, pints.PopulationBasedOptimiser):
                n_workers = min(n_workers, self._optimiser.population_size())
            evaluator = pints.ParallelEvaluator(f, n_workers=n_workers)
        else:
            evaluator = pints.SequentialEvaluator(f)

        # Keep track of best position and score
        xbest = self._optimiser._x0
        fbest = f(self._optimiser._x0) #float('inf')        
        print('fbest', fbest)

        # Internally we always minimise! Keep a 2nd value to show the user
        fbest_user = fbest if self._minimising else -fbest

        # Set up progress reporting
        next_message = 0

        # Start logging
        logging = self._log_to_screen or self._log_filename
        if logging:
            if self._log_to_screen:
                # Show direction
                if self._minimising:
                    print('Minimising error measure')
                else:
                    print('Maximising LogPDF')

                # Show method
                print('Using ' + str(self._optimiser.name()))

                # Show parallelisation
                if self._parallel:
                    print('Running in parallel with ' + str(n_workers) +
                          ' worker processes.')
                else:
                    print('Running in sequential mode.')

            # Show population size
            pop_size = 1
            if isinstance(self._optimiser, pints.PopulationBasedOptimiser):
                pop_size = self._optimiser.population_size()
                if self._log_to_screen:
                    print('Population size: ' + str(pop_size))

            # Set up logger
            logger = pints.Logger()
            if not self._log_to_screen:
                logger.set_stream(None)
            if self._log_filename:
                logger.set_filename(self._log_filename, csv=self._log_csv)

            # Add fields to log
            max_iter_guess = max(self._max_iterations or 0, 10000)
            max_eval_guess = max_iter_guess * pop_size
            logger.add_counter('Iter.', max_value=max_iter_guess)
            logger.add_counter('Eval.', max_value=max_eval_guess)
            logger.add_float('Best')
            self._optimiser._log_init(logger)
            logger.add_time('Time m:s')

        # Start searching
        timer = pints.Timer()
        running = True        
        nIter_update = 0        
        try:
            
            while running:
                # Get points
                xs = self._optimiser.ask()
                
                # Calculate scores
                fs = evaluator.evaluate(xs)
              
                # Perform iteration
                self._optimiser.tell(fs)

                # Check if new best found                
                fnew = self._optimiser.fbest()
                         
                if fnew < fbest:                    
                    # Check if this counts as a significant change
                    if np.abs(fnew - fbest) < self._min_significant_change:
                        unchanged_iterations += 1
                    else:
                        unchanged_iterations = 0

                    # Update best
                    xbest = self._optimiser.xbest()
                    fbest = fnew

                    # Update user value of fbest
                    fbest_user = fbest if self._minimising else -fbest
                    
                    nIter_update += 1       
                    self.update_record.append((iteration+1, nIter_update))
#                     print("Epoch : %d  |  nUpdate : %d"%(iteration+1, nIter_update))     
                else:
                    unchanged_iterations += 1
                    
                                
                # Update evaluation count
                evaluations += len(fs)

                # Show progress
                if logging and iteration >= next_message:
                    # Log state
                    logger.log(iteration, evaluations, fbest_user)
                    self._optimiser._log_write(logger)
                    logger.log(timer.time())

                    # Choose next logging point
                    if iteration < self._message_warm_up:
                        next_message = iteration + 1
                    else:
                        next_message = self._message_interval * (
                            1 + iteration // self._message_interval)

                # Update iteration count
                iteration += 1

                #
                # Check stopping criteria
                #

                # Maximum number of iterations
                if (self._max_iterations is not None and
                        iteration >= self._max_iterations):
                    running = False
                    halt_message = ('Halting: Maximum number of iterations ('
                                    + str(iteration) + ') reached.')

                # Maximum number of iterations without significant change
                halt = (self._max_unchanged_iterations is not None and
                        unchanged_iterations >= self._max_unchanged_iterations)
                if halt:
                    running = False
                    halt_message = ('Halting: No significant change for ' +
                                    str(unchanged_iterations) + ' iterations.')

                # Threshold value
                if self._threshold is not None and fbest < self._threshold:
                    running = False
                    halt_message = ('Halting: Objective function crossed'
                                    ' threshold: ' + str(self._threshold) +
                                    '.')

                # Error in optimiser
                error = self._optimiser.stop()
                if error:   # pragma: no cover
                    running = False
                    halt_message = ('Halting: ' + str(error))

                elif self._callback is not None:
                    self._callback(iteration - 1, self._optimiser)
                                       
        except (Exception, SystemExit, KeyboardInterrupt):  # pragma: no cover
            # Unexpected end!
            # Show last result and exit
            print('\n' + '-' * 40)
            print('Unexpected termination.')
            print('Current best score: ' + str(fbest))
            print('Current best position:')

            # Inverse transform search parameters
            if self._transformation:
                xbest = self._transformation.to_model(xbest)
#             else:
#                 xbest = xbest

            for p in xbest:
                print(pints.strfloat(p))
            print('-' * 40)
            raise

        # Stop timer
        self._time = timer.time()

        # Log final values and show halt message
        if logging:
            logger.log(iteration, evaluations, fbest_user)
            self._optimiser._log_write(logger)
            logger.log(self._time)
            if self._log_to_screen:
                print(halt_message)

        # Save post-run statistics
        self._evaluations = evaluations
        self._iterations = iteration

        # Inverse transform search parameters
        if self._transformation:
            xbest = self._transformation.to_model(xbest)
#         else:
#             xbest = xbest
        
        # Return best position and score
        return xbest, fbest_user


    def set_callback(self, cb=None):
        """
        Allows a "callback" function to be passed in that will be called at the
        end of every iteration.

        This can be used for e.g. visualising optimiser progress.

        Example::

            def cb(opt):
                plot(opt.xbest())

            opt.set_callback(cb)

        """
        if cb is not None and not callable(cb):
            raise ValueError('The argument cb must be None or a callable.')
        self._callback = cb


    def set_log_interval(self, iters=20, warm_up=3):
        """
        Changes the frequency with which messages are logged.

        Parameters
        ----------
        ``interval``
            A log message will be shown every ``iters`` iterations.
        ``warm_up``
            A log message will be shown every iteration, for the first
            ``warm_up`` iterations.
        """
        iters = int(iters)
        if iters < 1:
            raise ValueError('Interval must be greater than zero.')
        warm_up = max(0, int(warm_up))

        self._message_interval = iters
        self._message_warm_up = warm_up


    def set_log_to_file(self, filename=None, csv=False):
        """
        Enables logging to file when a filename is passed in, disables it if
        ``filename`` is ``False`` or ``None``.

        The argument ``csv`` can be set to ``True`` to write the file in comma
        separated value (CSV) format. By default, the file contents will be
        similar to the output on screen.
        """
        if filename:
            self._log_filename = str(filename)
            self._log_csv = True if csv else False
        else:
            self._log_filename = None
            self._log_csv = False


    def set_log_to_screen(self, enabled):
        """
        Enables or disables logging to screen.
        """
        self._log_to_screen = True if enabled else False


    def set_max_iterations(self, iterations=10000):
        """
        Adds a stopping criterion, allowing the routine to halt after the
        given number of ``iterations``.

        This criterion is enabled by default. To disable it, use
        ``set_max_iterations(None)``.
        """
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError(
                    'Maximum number of iterations cannot be negative.')
        self._max_iterations = iterations


    def set_max_unchanged_iterations(self, iterations=200, threshold=1e-11):
        """
        Adds a stopping criterion, allowing the routine to halt if the
        objective function doesn't change by more than ``threshold`` for the
        given number of ``iterations``.

        This criterion is enabled by default. To disable it, use
        ``set_max_unchanged_iterations(None)``.
        """
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError(
                    'Maximum number of iterations cannot be negative.')

        threshold = float(threshold)
        if threshold < 0:
            raise ValueError('Minimum significant change cannot be negative.')

        self._max_unchanged_iterations = iterations
        self._min_significant_change = threshold


    def set_parallel(self, parallel=False):
        """
        Enables/disables parallel evaluation.

        If ``parallel=True``, the method will run using a number of worker
        processes equal to the detected cpu core count. The number of workers
        can be set explicitly by setting ``parallel`` to an integer greater
        than 0.
        Parallelisation can be disabled by setting ``parallel`` to ``0`` or
        ``False``.
        """
        if parallel is True:
            self._parallel = True
            self._n_workers = pints.ParallelEvaluator.cpu_count()
        elif parallel >= 1:
            self._parallel = True
            self._n_workers = int(parallel)
        else:
            self._parallel = False
            self._n_workers = 1


    def set_threshold(self, threshold):
        """
        Adds a stopping criterion, allowing the routine to halt once the
        objective function goes below a set ``threshold``.

        This criterion is disabled by default, but can be enabled by calling
        this method with a valid ``threshold``. To disable it, use
        ``set_treshold(None)``.
        """
        if threshold is None:
            self._threshold = None
        else:
            self._threshold = float(threshold)


    def threshold(self):
        """
        Returns the threshold stopping criterion, or ``None`` if no threshold
        stopping criterion is set. See :meth:`set_threshold()`.
        """
        return self._threshold


    def time(self):
        """
        Returns the time needed for the last run, in seconds, or ``None`` if
        the controller hasn't run yet.
        """
        return self._time