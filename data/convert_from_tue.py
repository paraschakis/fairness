import pandas as pd

INPUT_FILE = 'raw/speed_dating_tue.csv'
OUTPUT_FILE = 'processed/speed_dating_from_tue.csv'

# parse the original file into a dataframe with 8378 rows and 123 columns
df = pd.read_csv(INPUT_FILE, sep=',')

# remove 105 records where 'rate yourself' features (e.g. 'attractive') are absent
# this also filters out all records where 'importance_same_race/religion' are absent
df = df[df.attractive.notnull()]

# =============================================================================
# remove columns that cannot be used for recommendations
# =============================================================================

# select columns with feedback provided after the meeting with a partner
# this information cannot be used for recommondations provided before the meeting
rating_by_partner = ['attractive_o', 'sinsere_o', 'intelligence_o', 'funny_o',
                     'ambitous_o', 'shared_interests_o']

rating_for_partner = ['attractive_partner', 'sincere_partner',
                      'intelligence_partner', 'funny_partner',
                      'ambition_partner', 'shared_interests_partner']

outcomes = ['like', 'guess_prob_liked', 'met', 'decision', 'decision_o']

# select discretized columns
discretized = ['d_d_age', 'd_importance_same_race', 'd_importance_same_religion',
               'd_pref_o_attractive', 'd_pref_o_sincere',
               'd_pref_o_intelligence', 'd_pref_o_funny', 'd_pref_o_ambitious',
               'd_pref_o_shared_interests', 'd_attractive_o', 'd_sinsere_o',
               'd_intelligence_o', 'd_funny_o', 'd_ambitous_o',
               'd_shared_interests_o', 'd_attractive_important',
               'd_sincere_important', 'd_intellicence_important',
               'd_funny_important', 'd_ambtition_important',
               'd_shared_interests_important', 'd_attractive', 'd_sincere',
               'd_intelligence', 'd_funny', 'd_ambition', 'd_attractive_partner',
               'd_sincere_partner', 'd_intelligence_partner', 'd_funny_partner',
               'd_ambition_partner', 'd_shared_interests_partner',
               'd_sports', 'd_tvsports', 'd_exercise', 'd_dining', 'd_museums',
               'd_art', 'd_hiking', 'd_gaming', 'd_clubbing', 'd_reading',
               'd_tv', 'd_theater', 'd_movies', 'd_concerts', 'd_music',
               'd_shopping', 'd_yoga', 'd_interests_correlate',
               'd_expected_happy_with_sd_people',
               'd_expected_num_interested_in_me', 'd_expected_num_matches',
               'd_like', 'd_guess_prob_liked']

# select other columns to remove
other = ['has_null', 'd_age']

df.drop([*rating_by_partner, *rating_for_partner, *other, *outcomes,
         *discretized], axis=1, inplace=True)


# =============================================================================
# create unique id for each participant (the ids are missing from the dataset)
# =============================================================================

# the ids are created from the combination of all user-related fields
df['id'] = df.apply(lambda row: hash((
        row.gender,
        row.age,
        row.race,
        row.importance_same_race,
        row.importance_same_religion,
        row.attractive_important,
        row.sincere_important,
        row.intellicence_important,
        row.funny_important,
        row.ambtition_important,
        row.shared_interests_important,
        row.attractive,
        row.sincere,
        row.intelligence,
        row.funny,
        row.ambition,
        row.sports,
        row.tvsports,
        row.exercise,
        row.dining,
        row.museums,
        row.art,
        row.hiking,
        row.gaming,
        row.clubbing,
        row.reading,
        row.tv,
        row.theater,
        row.movies,
        row.concerts,
        row.music,
        row.shopping,
        row.yoga,
        row.expected_happy_with_sd_people,
        row.expected_num_interested_in_me,
        row.expected_num_matches
        )), axis=1)

df.set_index('id', inplace=True)

# export to CSV
df.to_csv(OUTPUT_FILE, index_label='id')

# how many columns have missing values
na_cols = df.isna().any()
