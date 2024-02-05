from . import preprocessing as pp

class SpamDetector:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self._word_counts = None
        self.n_train_messages = None
        self.n_ham = None
        self.n_spam = None
        self._prob_info = None
    def fit(self, messages, classes):
        """Fit the model
        Parameters
        ----------
        messages: seq of str
            The train messages
        classes: seq of str
            The train classes
        Returns
        -------
        self: SpamDetector
            The fitted model
        """
        ann_msg = list(zip(classes, messages))  # Convert to list to be able to use `len`
        self.n_train_messages = len(ann_msg)
        self._word_counts = pp.count_words_in_ham_and_spam(ann_msg)
        self.n_ham, self.n_spam = pp.count_ham_and_spam(ann_msg)
        self._prob_info = estimate_probabilities(
            self._word_counts, (self.n_ham, self.n_spam), alpha=self.alpha
        )
        return self
    def predict_proba(self, messages):
        """Compute the predicted class probability
        Parameters
        ----------
        messages: str or seq of str
            A single message or a sequence of messages
        Returns
        -------
        proba: tuple or list of tuples
            For a single message it returns a tuple `(p_ham, p_spam)` for the message.
            It returns a list of such tuples if the input is a sequence of messages.
        """
        if isinstance(messages, str):  # Single message
            return classify_message(messages, self._prob_info)
        else:
            return [classify_message(msg, self._prob_info) for msg in messages]
    def predict(self, messages):
        """Compute the predicted class probability
        Parameters
        ----------
        messages: str or seq of str
            A single message or a sequence of messages
        Returns
        -------
        pred: str or list of str
            For a single message it returns the predicted class has a "ham" or "spam" string
            for the input message.
            It returns a list of predicted classes if the input is a sequence of messages.
        """
        probs = self.predict_proba(messages)
        if isinstance(messages, str):  # single message
            return "ham" if probs[0] > 0.5 else "spam"
        else:
            return [
                "ham" if p[0] > 0.5 else "spam"
                for p in probs
            ]

    def estimate_probabilities(word_counts, n_classes, alpha=0.0):
        """Estimate p(w|c) and p(c)
        Estimate conditional probability of words knowing the class
        as well as marginal probabilities based on frequencies.
        Parameters
        ----------
        word_counts: dict
            Dictionary of word counts provided as `(word, (n_w_ham, n_w_spam))` items.
        alpha: float, optional (default=0.0)
            The additive smoothing parameter.
        Returns
        -------
        prob_word_knowing_class:dict
            Probability of word knowing the class provided as `(word, (p_w_ham, p_w_spam))` items.
        prob_class: tuple
            Probability of class provided as a `(p_ham, p_spam)` tuple
        """
        n_ham, n_spam = n_classes
        n_tot = n_ham + n_spam + 2 * alpha
        if alpha > 0:
            n_ham += alpha
            n_spam += alpha
        p_ham = n_ham / n_tot
        p_spam = n_spam / n_tot
        p_class = (p_ham, p_spam)
        p_word_knowing_class = {}
        for word, (n_word_ham, n_word_spam) in word_counts.items():
            p_word_ham = (n_word_ham + alpha) / n_ham
            p_word_spam = (n_word_spam + alpha) / n_spam
            p_word_knowing_class[word] = (p_word_ham, p_word_spam)
        return p_word_knowing_class, p_class

    def classify_message(message, prob_info):
        """Compute the probability for a message to be ham or spam
        Parameters
        ----------
        message: str
            The input message.
        prob_info: tuple
            A tuple containing probability information computed on the training corpus as
            output by the `estimate_prob_word_knowing_class` function.
            It contains:
            p_word_knowing_class: dict
                The probability of a message to contain a given word knowing its class.
            p_class: tuple
                The probability of the class as a (p_ham, p_spam) tuple.
        Returns
        -------
        p_ham: float
            The probability of the message to belong to the ham class
        p_spam: float
            The probability of the message to belong to the spam class.
        """
        p_word_class, p_class = prob_info
        msg_words = pp.split_words(pp.clean_msg(message))
        p_msg_ham = 1.0
        p_msg_spam = 1.0
        for word in msg_words:
            p_word_ham, p_word_spam = p_word_class.get(word, (1.0, 1.0))
            p_msg_ham *= p_word_ham
            p_msg_spam *= p_word_spam
        p_ham, p_spam = p_class
        p_msg_ham_p_ham = p_msg_ham * p_ham
        p_msg_spam_p_spam = p_msg_spam * p_spam
        p_msg_tot = p_msg_ham_p_ham + p_msg_spam_p_spam
        return p_msg_ham_p_ham / p_msg_tot, p_msg_spam_p_spam / p_msg_tot

    def predict_word_proba(self, message):
            words = pp.split_words(pp.clean_msg(message))

            res = {}
            for w in words:
                if w in self._word_counts:
                    n_w_ham, n_w_spam = self._word_counts[w]

                    if self.alpha > 0.0:
                        n_w_ham += self.alpha
                        n_w_spam += self.alpha

                    n_w_tot = n_w_ham + n_w_spam
                    p_ham_word = n_w_ham / n_w_tot
                    p_spam_word = n_w_spam / n_w_tot
                    res[w] = (p_ham_word, p_spam_word)
            return res