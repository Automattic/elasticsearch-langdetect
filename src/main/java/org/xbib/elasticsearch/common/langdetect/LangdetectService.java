package org.xbib.elasticsearch.common.langdetect;

import org.elasticsearch.ElasticsearchException;
import org.elasticsearch.common.logging.ESLogger;
import org.elasticsearch.common.logging.ESLoggerFactory;
import org.elasticsearch.common.settings.Settings;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.regex.Pattern;

public class LangdetectService {
    private final static long RANDOM_SEED = 0;
    private final static ESLogger logger = ESLoggerFactory.getLogger(LangdetectService.class.getName());
    private final Settings settings;
    private final static Pattern word = Pattern.compile("[\\P{IsWord}]", Pattern.UNICODE_CHARACTER_CLASS);
    public final static String[] DEFAULT_LANGUAGES = new String[] {
            "ar",
            "bg",
            "bn",
            "cs",
            "da",
            "de",
            "el",
            "en",
            "es",
            "et",
            "fa",
            "fi",
            "fr",
            "gu",
            "he",
            "hi",
            "hr",
            "hu",
            "id",
            "it",
            "ja",
            "ko",
            "lt",
            "lv",
            "mk",
            "ml",
            "nl",
            "no",
            "pa",
            "pl",
            "pt",
            "ro",
            "ru",
            "sq",
            "sv",
            "ta",
            "te",
            "th",
            "tl",
            "tr",
            "uk",
            "ur",
            "vi",
            "zh-cn",
            "zh-tw"
    };
    private final static Settings DEFAULT_SETTINGS = Settings.builder()
            .putArray("languages", DEFAULT_LANGUAGES)
            .build();

    private Map<String, double[]> wordLangProbMap = new HashMap<>();
    private List<String> langlist = new LinkedList<>();
    private Map<String,String> langmap = new HashMap<>();

    private final String profileParam;
    private final double alpha;
    private final double alphaWidth;
    private final int numTrials;
    private final int iterationLimit;
    private final double probThreshold;
    private final double convThreshold;
    private final int baseFreq;
    private final Pattern filterPattern;

    /**
     * Create a service with the default settings.
     */
    public LangdetectService() {
        this(DEFAULT_SETTINGS);
    }

    /**
     * Create a service with the given settings and the default language profile. 
     */
    public LangdetectService(Settings settings) {
        this(settings, null);
    }

    /**
     * Create a service with the given settings and language profile (null or "short-text").
     */
    public LangdetectService(Settings settings, String profileParam) {
        this.settings = settings;
        this.profileParam = settings.get("profile", profileParam);
        load(settings);
        this.numTrials = settings.getAsInt("number_of_trials", 7);
        this.alpha = settings.getAsDouble("alpha", 0.5);
        this.alphaWidth = settings.getAsDouble("alpha_width", 0.05);
        this.iterationLimit = settings.getAsInt("iteration_limit", 10000);
        this.probThreshold = settings.getAsDouble("prob_threshold", 0.1);
        this.convThreshold = settings.getAsDouble("conv_threshold",  0.99999);
        this.baseFreq = settings.getAsInt("base_freq", 10000);
        this.filterPattern = settings.get("pattern") != null ?
                                 Pattern.compile(settings.get("pattern"), Pattern.UNICODE_CHARACTER_CLASS) : null;
    }

    /**
     * Return the settings used to create this service. 
     */
    public Settings getSettings() {
        return settings;
    }

    /**
     * Populate this service's fields according to the given settings.  
     */
    private void load(Settings settings) {
        if (settings.equals(Settings.EMPTY)) {
            return;
        }
        try {
            String[] keys = DEFAULT_LANGUAGES;
            if (settings.get("languages") != null) {
                keys = settings.get("languages").split(",");
            }
            List<LangProfile> langProfiles = new ArrayList<>();
            for (String key : keys) {
                if (key != null && !key.isEmpty()) {
                    langProfiles.addAll(loadProfilesFromResource(key));
                }
            }
            for (int i = 0; i < langProfiles.size(); i++) {
                addProfile(langProfiles.get(i), i, langProfiles.size());
            }
            logger.debug("language detection service installed for {}", langlist);
        } catch (Exception e) {
            logger.error(e.getMessage(), e);
            throw new ElasticsearchException(Arrays.toString(e.getStackTrace()) + "\n" + e.getMessage() +
                                             " profile=" + profileParam);
        }
        try {
            // map by settings
            Settings map = Settings.EMPTY;
            if (settings.getByPrefix("map.") != null) {
                map = Settings.settingsBuilder().put(settings.getByPrefix("map.")).build();
            }
            if (map.getAsMap().isEmpty()) {
                // is in "map" a resource name?
                String mapResource = settings.get("map");
                if (mapResource == null) {
                    mapResource = "language.json";
                }
                InputStream in = getClass().getResourceAsStream(mapResource);
                if (in != null) {
                    map = Settings.settingsBuilder().loadFromStream(mapResource, in).build();
                }
            }
            this.langmap = map.getAsMap();
        } catch (Exception e) {
            logger.error(e.getMessage(), e);
            throw new ElasticsearchException(e.getMessage());
        }
    }

    /**
     * Load language profiles from resource files for the given language.
     */
    private List<LangProfile> loadProfilesFromResource(String lang) throws IOException {
        List<String> profilePaths = new ArrayList<>();
        if (profileParam == null) {
            profilePaths.add("/langdetect/" + lang);
        } else {
            for (String prof : profileParam.split(",")) {
                profilePaths.add("/langdetect/" + prof + "/" + lang);
            }
        }
        List<LangProfile> langProfiles = new ArrayList<>();
        for (String profilePath : profilePaths) {
            InputStream in = getClass().getResourceAsStream(profilePath);
            if (in == null) {
                if (profilePaths.size() == 1) {
                    throw new IOException("profile '" + lang + "' not found");
                } else {
                    logger.warn("profile '" + lang + "' not found in path " + profilePath);
                    continue;
                }
            }
            langProfiles.add(new LangProfile(in));
        }
        return langProfiles;
    }

    /**
     * Add a language profile to this service.
     *
     * Note: This method should probably not be public as it requires callers to know the inner workings of this class.
     *       Use at your own risk!
     */
    public void addProfile(LangProfile profile, int profileIndex, int numLanguages) throws IOException {
        String lang = profile.getName();
        langlist.add(lang);
        List<Long> profileNWords = profile.getNWords();
        Map<String, Long> oneSkipBigramFreqs = new HashMap<>();
        Set<String> changedWords = new HashSet<>();
        long oneSkipBigramNWords = 0;
        for (Map.Entry<String, Long> entry : profile.getFreq().entrySet()) {
            String word = entry.getKey();
            if ("lowercase".equals(settings.get("experimentName"))) {
                word = word.toLowerCase(Locale.ROOT);
            }
            Long wordCount = entry.getValue();
            int len = word.length();
            if (len < 1 || len > NGram.N_GRAM) {
                continue;
            }
            if (!wordLangProbMap.containsKey(word)) {
                wordLangProbMap.put(word, new double[numLanguages]);
            }
            wordLangProbMap.get(word)[profileIndex] += wordCount.doubleValue();
            changedWords.add(word);
            if ("one-skip-bigrams".equals(settings.get("experimentName")) && len == 3) {
                String oneSkipBigram = "1sb:" + word.charAt(0) + word.charAt(2);
                oneSkipBigramNWords += wordCount;
                if (oneSkipBigramFreqs.containsKey(oneSkipBigram)) {
                    oneSkipBigramFreqs.put(oneSkipBigram, oneSkipBigramFreqs.get(oneSkipBigram) + wordCount);
                } else {
                    oneSkipBigramFreqs.put(oneSkipBigram, wordCount);
                }
            }
        }
        for (String word : changedWords) {
            wordLangProbMap.get(word)[profileIndex] /= profileNWords.get(word.length() - 1);
        }
        if ("one-skip-bigrams".equals(settings.get("experimentName"))) {
            for (Map.Entry<String, Long> entry : oneSkipBigramFreqs.entrySet()) {
                String oneSkipBigram = entry.getKey();
                if (!wordLangProbMap.containsKey(oneSkipBigram)) {
                    wordLangProbMap.put(oneSkipBigram, new double[numLanguages]);
                }
                wordLangProbMap.get(oneSkipBigram)[profileIndex] = entry.getValue().doubleValue() / oneSkipBigramNWords;
            }
        }
    }

    /**
     * Detect the languages in the text, returning a list of languages sorted in descending order of probability.
     */
    public List<Language> detectAll(String text) throws LanguageDetectionException {
        text = NGram.normalizeVietnamese(text);
        if (filterPattern != null && !filterPattern.matcher(text).matches()) {
            return Collections.emptyList();
        }
        List<Language> languages = convertProbabilitiesToLanguages(detectProbabilities(text));
        return languages.subList(0, Math.min(languages.size(), settings.getAsInt("max", languages.size())));
    }

    /**
     * Return an array representing the probability distribution of the text's language.
     */
    private double[] detectProbabilities(String text) throws LanguageDetectionException {
        // clean all non-word characters from text
        text = text.replaceAll(word.pattern(), " ");
        List<String> ngrams = extractNGrams(text);
        double[] overallProbs = new double[langlist.size()];
        if (ngrams.isEmpty()) {
            return overallProbs;
        }
        Random rand = new Random(RANDOM_SEED);
        for (int t = 0; t < numTrials; ++t) {
            double[] trialProbs = new double[langlist.size()];
            Arrays.fill(trialProbs, 1.0 / langlist.size());
            double weight = (alpha + rand.nextGaussian() * alphaWidth) / baseFreq;
            if (Objects.equals(settings.get("experimentName"), "no-ngram-subsampling")) {
                for (int i = 0; i < ngrams.size(); i++) {
                    double[] langProbMap = wordLangProbMap.get(ngrams.get(i));
                    for (int j = 0; j < trialProbs.length; ++j) {
                        trialProbs[j] *= weight + langProbMap[j];
                    }
                    // Normalize every few iterations to avoid over/underflow
                    if (i % 5 == 0) {
                        normalizeProbabilities(trialProbs);
                    }
                }
                normalizeProbabilities(trialProbs);
            } else {
                for (int i = 0; ; ++i) {
                    double[] langProbMap = wordLangProbMap.get(ngrams.get(rand.nextInt(ngrams.size())));
                    for (int j = 0; j < trialProbs.length; ++j) {
                        trialProbs[j] *= weight + langProbMap[j];
                    }
                    if (i % 5 == 0 && (normalizeProbabilities(trialProbs) > convThreshold || i >= iterationLimit)) {
                        break;
                    }
                }
            }
            for (int j = 0; j < overallProbs.length; ++j) {
                overallProbs[j] += trialProbs[j] / numTrials;
            }
        }
        return overallProbs;
    }

    /**
     * Convert the text to a list of ngrams of length 1 to NGram.N_GRAM (unseen ngrams are omitted). 
     */
    private List<String> extractNGrams(String text) {
        List<String> ngrams = new ArrayList<>();
        NGram ngramGenerator = new NGram();
        if ("lowercase".equals(settings.get("experimentName"))) {
            text = text.toLowerCase(Locale.ROOT);
        }
        for (int i = 0; i < text.length(); ++i) {
            ngramGenerator.addChar(text.charAt(i));
            for (int n = 1; n <= NGram.N_GRAM; ++n) {
                String ngram = ngramGenerator.get(n);
                if (ngram == null) {
                    continue;
                }
                if (wordLangProbMap.containsKey(ngram)) {
                    ngrams.add(ngram);
                }
                if ("one-skip-bigrams".equals(settings.get("experimentName")) && ngram.length() == 3) {
                    String oneSkipBigram = "1sb:" + ngram.charAt(0) + ngram.charAt(2);
                    if (wordLangProbMap.containsKey(oneSkipBigram)) {
                        ngrams.add(oneSkipBigram);
                    }
                }
            }
        }
        return ngrams;
    }

    /**
     * Normalize the array of probabilities so that they sum to one, returning the maximum normalized value.
     */
    private double normalizeProbabilities(double[] probs) {
        double max = 0;
        double sum = 0;
        for (double prob : probs) {
            sum += prob;
        }
        for (int i = 0; i < probs.length; ++i) {
            double p = probs[i] / sum;
            if (max < p) {
                max = p;
            }
            probs[i] = p;
        }
        return max;
    }

    /**
     * Convert the array of probabilities to a list of Langauge objects, sorted in descending order of probability.
     */
    private List<Language> convertProbabilitiesToLanguages(double[] probs) {
        List<Language> languages = new ArrayList<>();
        for (int j = 0; j < probs.length; ++j) {
            double p = probs[j];
            if (p > probThreshold) {
                String code = langlist.get(j);
                languages.add(new Language(langmap != null && langmap.containsKey(code) ? langmap.get(code) : code, p));
            }
        }
        Collections.sort(languages, new Comparator<Language>() {
            @Override
            public int compare(Language l1, Language l2) {
                return Double.compare(l2.getProbability(), l1.getProbability());
            }
        });
        return languages;
    }
}
