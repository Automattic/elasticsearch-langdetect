package org.xbib.elasticsearch.index.mapper.langdetect;

import com.google.common.base.Joiner;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.common.settings.Settings;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.xbib.elasticsearch.common.langdetect.LangdetectService;
import org.xbib.elasticsearch.common.langdetect.Language;
import org.xbib.elasticsearch.common.langdetect.LanguageDetectionException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeSet;

@RunWith(Parameterized.class)
public class DetectLanguageAccuracyTest extends Assert {
    private static final Logger logger = LogManager.getLogger();

    private static final double ACCURACY_DELTA = 1e-6;
    private static final String ALL_LANGUAGES =
        "af,ar,bg,bn,ca,cs,da,de,el,en,es,et,fa,fi,fr,gu,he,hi,hr,hu,id,it,ja,kn,ko,lt,lv,mk,ml,mr,ne,nl,no,pa,pl,pt," +
            "ro,ru,si,sk,sl,so,sq,sv,sw,ta,te,th,tl,tr,uk,ur,vi,zh-cn,zh-tw";
    private static final String DEFAULT_LANGUAGES = Joiner.on(",").join(LangdetectService.DEFAULT_LANGUAGES);
    private static final String ALL_DEFAULT_PROFILE_LANGUAGES =
        "af,ar,bg,bn,cs,da,de,el,en,es,et,fa,fi,fr,gu,he,hi,hr,hu,id,it,ja,kn,ko,lt,lv,mk,ml,mr,ne,nl,no,pa,pl,pt,ro," +
            "ru,sk,sl,so,sq,sv,sw,ta,te,th,tl,tr,uk,ur,vi,zh-cn,zh-tw";
    private static final String ALL_SHORT_PROFILE_LANGUAGES =
        "ar,bg,bn,ca,cs,da,de,el,en,es,et,fa,fi,fr,gu,he,hi,hr,hu,id,it,ja,ko,lt,lv,mk,ml,nl,no,pa,pl,pt,ro,ru,si,sq," +
            "sv,ta,te,th,tl,tr,uk,ur,vi,zh-cn,zh-tw";
    private static final Set<String> MERGED_AVERAGE_ONLY_EXPERIMENTS = new HashSet<>(
        Arrays.asList("one-skip-bigrams", "lowercase", "ensemble-lowercase", "confusion-matrix")
    );

    private static Map<String, Map<String, List<String>>> multiLanguageDatasets;
    private static Path outputPath;

    private final String datasetName;
    private final int substringLength;
    private final int sampleSize;
    private final String profileParam;
    private final boolean useAllLanguages;
    private final Map<String, Double> languageToExpectedAccuracy;

    /**
     * Construct a test for classification accuracies on substrings of texts from a single dataset.
     *
     * For each text and substring length, this test generates a sample of substrings (drawn uniformly with
     * replacement from the set of possible substrings of the given length), runs the language identification code,
     * measures the per-language accuracy (percentage of substrings classified correctly), and fails if the accuracy
     * varies by more than {@link #ACCURACY_DELTA} from the expected accuracy for the language.
     *
     * @param datasetName multi-language dataset name, as read in the setup step (see {@link #setUp()})
     * @param substringLength substring length to test (see {@link #generateSubstringSample(String, int, int)})
     * @param sampleSize number of substrings to test (see {@link #generateSubstringSample(String, int, int)})
     * @param profileParam profile name parameter to pass to the detection service 
     * @param useAllLanguages if true, all supported languages will be used instead of  just the default ones
     * @param languageToExpectedAccuracy mapping from language code to expected accuracy 
     */
    public DetectLanguageAccuracyTest(String datasetName,
                                      int substringLength,
                                      int sampleSize,
                                      String profileParam,
                                      boolean useAllLanguages,
                                      Map<String, Double> languageToExpectedAccuracy) {
        this.datasetName = datasetName;
        this.substringLength = substringLength;
        this.sampleSize = sampleSize;
        this.profileParam = profileParam;
        this.useAllLanguages = useAllLanguages;
        this.languageToExpectedAccuracy = languageToExpectedAccuracy;
    }

    /**
     * Perform the common set up tasks for tests of this class: read the datasets, and write the header row of the
     * output CSV if the path.accuracies.out system property is set.
     */
    @BeforeClass
    public static void setUp() throws IOException {
        multiLanguageDatasets = new HashMap<>();
        multiLanguageDatasets.put("udhr", readMultiLanguageDataset("udhr.tsv"));
        multiLanguageDatasets.put("wordpress-translations", readMultiLanguageDataset("wordpress-translations.tsv"));

        String outputPathStr = System.getProperty("path.accuracies.out");
        if (outputPathStr != null && !outputPathStr.isEmpty()) {
            logger.warn("File argument given ({}) -- running in output mode without assertions", outputPathStr);
            outputPath = Paths.get(outputPathStr);
            // Write column headers
            Files.write(
                outputPath,
                Collections.singletonList(
                    "datasetName,substringLength,sampleSize,profileParam,useAllLanguages," +
                    ("confusion-matrix".equals(System.getProperty("experiment.name")) ? "actual,null," : "") +
                    ALL_LANGUAGES
                ),
                StandardCharsets.UTF_8
            );
        }
    }

    /**
     * Run the test according to the parameters passed to the constructor.
     *
     * If {@link #outputPath} is not null, the test always passes and the results are written to the output path.
     */
    @Test
    public void test() throws IOException {
        // Set up the detection service according to the test's parameters
        String languageSetting = DEFAULT_LANGUAGES;
        if (useAllLanguages) {
            // TODO: This is a bit clunky. LangdetectService should support "all" as a language setting.
            if (profileParam.isEmpty()) {
                languageSetting = ALL_DEFAULT_PROFILE_LANGUAGES;
            } else if (profileParam.equals("short-text")) {
                languageSetting = ALL_SHORT_PROFILE_LANGUAGES;
            } else {
                assertEquals(profileParam, "merged-average");
                languageSetting = ALL_LANGUAGES;
            }
        }
        String profileOverride = profileParam;
        String experimentName = System.getProperty("experiment.name");
        if (experimentName != null && experimentName.startsWith("profile-")) {
            profileOverride = experimentName.split("-", 2)[1];
            // No point running experiments on the non-default profile. Experiments on the full union of languages are
            // left for future work.
            if (!profileParam.isEmpty() || useAllLanguages) {
                return;
            }
        }
        if (Objects.equals(experimentName, "ensemble-profiles") && (!profileParam.isEmpty() || useAllLanguages)) {
            return;
        }
        if (MERGED_AVERAGE_ONLY_EXPERIMENTS.contains(experimentName) &&
            (!profileParam.equals("merged-average") || !useAllLanguages)) {
            return;
        }
        LangdetectService service = createService(languageSetting, profileOverride, experimentName);
        Map<String, List<String>> languageToFullTexts = multiLanguageDatasets.get(datasetName);
        Set<String> testedLanguages = new TreeSet<>(languageToFullTexts.keySet());
        testedLanguages.retainAll(Arrays.asList(service.getSettings().getAsArray("languages")));

        // Classify the texts and calculate the accuracy for each language 
        Map<String, Double> languageToAccuracy = new HashMap<>(testedLanguages.size());
        for (String language : testedLanguages) {
            Map<String, Integer> predictedLanguageCounts = new HashMap<>();
            double numCorrect = 0;
            List<String> fullTexts = languageToFullTexts.get(language);
            for (String text : fullTexts) {
                for (String substring : generateSubstringSample(text, substringLength, sampleSize)) {
                    String predictedLanguage = DetectLanguageTest.getTopLanguageCode(service, substring);
                    if (Objects.equals(predictedLanguage, language)) {
                        numCorrect++;
                    }
                    if ("confusion-matrix".equals(experimentName)) {
                        if (predictedLanguageCounts.containsKey(predictedLanguage)) {
                            predictedLanguageCounts.put(predictedLanguage,
                                                        predictedLanguageCounts.get(predictedLanguage) + 1);
                        } else {
                            predictedLanguageCounts.put(predictedLanguage, 1);
                        }
                    }
                }
            }
            double accuracy = numCorrect / (fullTexts.size() * sampleSize);
            languageToAccuracy.put(language, accuracy);
            logger.debug("Language: {} Accuracy: {}", language, accuracy);
            if ("confusion-matrix".equals(experimentName)) {
                List<Object> row = new ArrayList<>();
                Collections.addAll(row, datasetName, substringLength, sampleSize, profileParam, useAllLanguages,
                                   language);
                row.add(predictedLanguageCounts.containsKey(null) ? predictedLanguageCounts.get(null) : 0);
                for (String predictedLanguage : ALL_LANGUAGES.split(",")) {
                    row.add(predictedLanguageCounts.containsKey(predictedLanguage) ?
                            predictedLanguageCounts.get(predictedLanguage) : 0);
                }
                Files.write(outputPath,
                            Collections.singletonList(Joiner.on(",").join(row)),
                            StandardCharsets.UTF_8,
                            StandardOpenOption.APPEND);
            }
        }

        // If no output file is given, compare the obtained accuracies to the expected values. Otherwise, write the
        // results to the output path without any assertions.
        if (outputPath == null) {
            assertEquals(languageToExpectedAccuracy.size(), languageToAccuracy.size());
            for (Map.Entry<String, Double> entry : languageToAccuracy.entrySet()) {
                assertEquals(languageToExpectedAccuracy.get(entry.getKey()), entry.getValue(), ACCURACY_DELTA);
            }
        } else if (!"confusion-matrix".equals(experimentName)) {
            List<Object> row = new ArrayList<>();
            Collections.addAll(row, datasetName, substringLength, sampleSize, profileParam, useAllLanguages);
            for (String language : ALL_LANGUAGES.split(",")) {
                row.add(languageToAccuracy.containsKey(language) ? languageToAccuracy.get(language) : Double.NaN);
            }
            Files.write(outputPath,
                        Collections.singletonList(Joiner.on(",").join(row)),
                        StandardCharsets.UTF_8,
                        StandardOpenOption.APPEND);
        }
    }

    /**
     * Read and parse the test parameters from the accuracies.csv resource.
     *
     * @return the parsed parameters
     */
    @Parameterized.Parameters(name="{0}: substringLength={1} sampleSize={2} profileParam={3} useAllLanguages={4}")
    public static Collection<Object[]> data() throws IOException {
        List<Object[]> data = new ArrayList<>();
        try (BufferedReader br = getResourceReader("accuracies.csv")) {
            // Skip header line
            br.readLine();
            String line;
            while ((line = br.readLine()) != null) {
                Scanner scanner = new Scanner(line).useDelimiter(",");
                data.add(new Object[] {
                    // datasetName
                    scanner.next(),
                    // substringLength
                    scanner.nextInt(),
                    // sampleSize
                    scanner.nextInt(),
                    // profileParam
                    scanner.next(),
                    // useAllLanguages
                    scanner.nextBoolean(),
                    // languageToExpectedAccuracy
                    null
                });
                Map<String, Double> languageToExpectedAccuracy = new HashMap<>();
                for (String language : ALL_LANGUAGES.split(",")) {
                    double expectedAccuracy = scanner.nextDouble();
                    if (!Double.isNaN(expectedAccuracy)) {
                        languageToExpectedAccuracy.put(language, expectedAccuracy);
                    }
                }
                data.get(data.size() - 1)[5] = languageToExpectedAccuracy;
            }
        }
        return data;
    }

    /**
     * Read and parse a multi-language dataset from the given path.
     *
     * @param path resource path, where the file is in tab-separated format with two columns: language code and text
     * @return a mapping from each language code found in the file to the texts of this language    
     */
    private static Map<String, List<String>> readMultiLanguageDataset(String path) throws IOException {
        Map<String, List<String>> languageToFullTexts = new HashMap<>();
        try (BufferedReader br = getResourceReader(path)) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] splitLine = line.split("\t");
                String language = splitLine[0];
                if (!languageToFullTexts.containsKey(language)) {
                    languageToFullTexts.put(language, new ArrayList<String>());
                }
                languageToFullTexts.get(language).add(splitLine[1]);
            }
        }
        return languageToFullTexts;
    }

    /**
     * Helper method to open a resource path and return it as a BufferedReader instance.
     */
    private static BufferedReader getResourceReader(String path) throws IOException {
        return new BufferedReader(new InputStreamReader(DetectLanguageAccuracyTest.class.getResourceAsStream(path),
                                                        StandardCharsets.UTF_8));
    }

    /**
     * Generate a random sample of substrings from the given text.
     *
     * Sampling is performed uniformly with replacement from the set of substrings of the provided text, ignoring
     * whitespace-only substrings. The random seed is set to a deterministic function of the method's parameters, so
     * repeated calls to this method with the same parameters will return the same sample.
     *
     * @param text the text from which the substring sample is drawn
     * @param substringLength length of each generated substring (set to zero to return a singleton list with the
     *                        text -- sampleSize must be 1 in this case)
     * @param sampleSize number of substrings to include in the sample
     * @return the sample (a list of strings)
     */
    private List<String> generateSubstringSample(String text, int substringLength, int sampleSize) {
        if (substringLength == 0 && sampleSize == 1) {
            return Collections.singletonList(text);
        }
        if (substringLength > text.trim().length()) {
            throw new IllegalArgumentException("Provided text is too short.");
        }
        Random rnd = new Random(Objects.hash(text, substringLength, sampleSize));
        List<String> sample = new ArrayList<>(sampleSize);
        while (sample.size() < sampleSize) {
            int startIndex = rnd.nextInt(text.length() - substringLength + 1);
            String substring = text.substring(startIndex, startIndex + substringLength);
            if (!substring.trim().isEmpty()) {
                sample.add(substring);
            }
        }
        return sample;
    }

    private LangdetectService createService(String languageSetting, String profileOverride, String experimentName) {
        Settings.Builder settingsBuilder = Settings.builder().put("languages", languageSetting)
                                                             .put("profile", profileOverride)
                                                             .put("experimentName", experimentName);
        if (!"ensemble-profiles".equals(experimentName) && !"ensemble-lowercase".equals(experimentName)) {
            return new LangdetectService(settingsBuilder.build());
        }
        settingsBuilder.put("prob_threshold", -1.0);
        Settings mainSettings;
        Settings otherSettings;
        if ("ensemble-profiles".equals(experimentName)) {
            mainSettings = settingsBuilder.put("profile", "").build();
            otherSettings = settingsBuilder.put("profile", "short-text").build();
        } else {
            mainSettings = settingsBuilder.put("experimentName", "").build();
            otherSettings = settingsBuilder.put("experimentName", "lowercase").build();
        }
        final LangdetectService otherService = new LangdetectService(otherSettings);
        return new LangdetectService(mainSettings) {
            @Override
            public List<Language> detectAll(String text) throws LanguageDetectionException {
                Map<String, Double> otherServiceProbabilities = new HashMap<>();
                for (Language language : otherService.detectAll(text)) {
                    otherServiceProbabilities.put(language.getLanguage(), language.getProbability());
                }
                List<Language> languages = new ArrayList<>();
                for (Language language : super.detectAll(text)) {
                    languages.add(
                        new Language(
                            language.getLanguage(),
                            (language.getProbability() + otherServiceProbabilities.get(language.getLanguage())) / 2
                        )
                    );
                }
                Collections.sort(languages, new Comparator<Language>() {
                    @Override
                    public int compare(Language l1, Language l2) {
                        return Double.compare(l2.getProbability(), l1.getProbability());
                    }
                });
                return languages;
            }
        };
    }
}
