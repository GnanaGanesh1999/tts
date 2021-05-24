import sys

def setup():
    # The path to the local git repo for Indic NLP library
    INDIC_NLP_LIB_HOME = r"E:\Final Year Project\packages\indic_nlp_library"

    # The path to the local git repo for Indic NLP Resources
    INDIC_NLP_RESOURCES = r"E:\Final Year Project\packages\indic_nlp_resources"

    # Add library to Python path
    sys.path.append(r'{}\src'.format(INDIC_NLP_LIB_HOME))

    # Set environment variable for resources folder
    # common.set_resources_path(INDIC_NLP_RESOURCES)


setup()