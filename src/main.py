from llm import LLM

if __name__ == '__main__':
    old_method = ('private String getSessionFileName(String sessionIdentifier , boolean createSessionFolder) {File '
                  'sessionFolder = folders.get(sessionIdentifier , createSessionFolder); return new File(sessionFolder , '
                  '" data" ). getAbsolutePath (); }')
    new_method = ('private String getSessionFileName(String sessionIdentifier) {File sessionFolder = folders.get('
                  'sessionIdentifier , true); return new File(sessionFolder , " data" ). getAbsolutePath (); }')
    old_comment = (
        'If the session folder (folder that contains the ﬁle) does not exist and createSessionFolder is true, '
        'the folder will be created.')

    llm = LLM('llama2')
    print(llm.infer_new_comment(old_comment, new_method, old_comment))
