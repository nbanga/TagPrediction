from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import string
from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

# converts text to stems
def tokenize(text):
    text = text.lower()
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if not w in stopwords.words('english')]
    stems = stem_tokens(filtered, stemmer)
    return stems

# converts text to pos_tags
def pos_tokens(text):
    stems = tokenize(text)
    list_of_stems = list(map(lambda x: [x], stems))
    result = [val for sublist in list_of_stems for val in pos_tag(sublist)]
    return result

# Counter of type of pos_tags in text
def freq_pos(text):
    pos_tags = pos_tokens(text)
    pos = [tags[1] for tags in pos_tags]
    count = Counter(pos)
    return count
    
# convert text to filtered tokens with pos_tag in "NN,JJ,NNS"
def filtered_tokens(text):
    pos_tags = pos_tokens(text)
    result = []
    for each in pos_tags:
        if each[1] in ['NN', 'JJ', 'NNS']:
            result.append(each[0])
    return result

	
file_path = "../../Data/data/all_labels.json"
label_file = open(file_path,'r')
labels = label_file.read()
#print(freq_pos(labels).most_common(100))
label_file.close()	
	
text = "c# c++ asp.net .net objective-c ruby-on-rails sql-server windows-7 \
asp.net-mvc vb.net ruby-on-rails-3 visual-studio-2010 asp.net-mvc-3 \
visual-studio web-services sql-server-2008 actionscript-3 cocoa-touch \
entity-framework internet-explorer jquery-ui .htaccess google-chrome \
node.js windows-xp unit-testing"

text = "c# asp.net c++ .net  c a b k for int i=0;i<n;i++ \
build onresume upload file-upload php file xsp4"

text = """"intel-rst": 41894, "httppostedfilebase": 32628, "datatemplates": 41896, "splines": 41897, "linkpoint": 42015, "git-submodules": 41899, "public-domain": 41900, "android-event": 41901, "dynamic-cast": 41902, "socket-server": 41903, "iphone-private-api": 41904, "interactive-login": 41905, "burst-mode": 41906, "68000": 38392, "java-deployment-toolkit": 41908, "sequence": 41308, "settext": 41910, "searching": 41911, "jscalendar": 41912, "110": 41913, "django-uploads": 41914, "sailfish-os": 41915, "cfthread": 41916, "lead": 41917, "cocoalibspotify-2.0": 41918, "leak": 41919, "dynamically-generated": 31745, "cross-domain": 41921, "domain-registrar": 41922, "page-caching": 41923, "table-per-hierachy": 41924, "artifactory": 13612, "locate": 41926, "11g": 41927, "fastinfoset": 41928, "enum": 41929, "character-replacement": 41930, "mitk": 41931, "app-tabs": 41932, "nscountedset": 21956, "sysinfo": 41934, "slug": 41935, "statspack": 4069, "intel-celeron": 41937, "shipping": 41497, "strassen": 21767, "nscache": 41940, "winddk": 41941, "oracle-adf": 41942, "exposure-correction": 41521, "pynotify": 41944, "development-fabric": 42027, "warranty": 41946, "uniform-integrability": 41947, "pointer-arithmetic": 41948, "apache-fop": 41949, "libtiff": 41950, "getschematable": 41951, "umbraco": 41952, "scrummaster": 41953, "brush": 41954, "registration": 28093, "page-zoom": 41956, "indefinite-integral": 41957, "manageddirectx": 41958, "editing-menu": 36918, "memory-access": 41959, "secd-machine": 41960, "wii-u": 41961, "opennebula": 41962, "network-map": 41963, "google-data-api": 41964, "mozy": 41965, "physx": 41966, "video-camera": 41967, "log4net": 19933, "gwt-2.3": 41969, "multigrid": 41970, "gwt-2.4": 41971, "gwt-2.5": 41972, "text-driver": 41973, "libmagic": 41974, "compaq-visual-fortran": 41975, "blobstorage": 41976, "mobile-controls": 41977, "3d-reconstruction": 41978, "ixmldomelement": 41979, "regexkitlite": 41715, "mashery": 41981, "siteedit": 41982, "lvs": 41983, "projective-geometry": 41984, "ghost4j": 41985, "icollection": 20341, "default-implementation": 41987, "unary-languages": 41988, "tactionmanager": 41989, "lvm": 41990, "backport": 34670, "gaps-and-islands": 41992, "metric-system": 41993, "php-5.2": 41994, "ppmcolormask": 41995, "install-name-tool": 41996, "prerender": 27729, "three-tier": 41998, "partially-applied-type": 32789, "dotnet-framework": 42000, "robospice": 42001, "context-sensitive-help": 42002, "jmdns": 9186, "configurationsection": 42004, "freelancing": 42005, "sessioncontext": 42006, "xsbt-web-plugin": 42007, "percent": 41857, "sicp": 41859, "peoplesoft": 42010, "mcapi": 41246, "book": 41876, "branch": 42013, "theos": 42014, "memoize": 41898, "console-redirect": 42016, "perl-xs": 42017, "table-partitioning": 42018, "interop-domino": 42019, "mobile-application": 42020, "snapstodevicepixels": 42021, "week-number": 42022, "jung": 42023, "xmldataset": 42024, "inappstorewindow": 42025, "lens-construction": 42026, "asset-pipeline": 41945, "fiddler2": 42028, "contiki": 17580, "ifilter": 4762, "predicate-logic": 42031, "autoversioning": 42032, "tracekit": 42033, "jscript": 42034, "space": 2882, "geometric-construction": 16442, "delphi-3": 42037, "setupapi": 42038, "jstorage": 42039, "moderncv": 37343, "xsp4": 42041, "mac-office": 42042, "gkmatchmaker": 42044, "vim-plugins": 42045, "numeric-conversion": 42046, "urchin": 42047, "uncle-bob": 42043"""

text = """How to check if an uploaded file is an image without mime type?","<p>I'd like to check if an uploaded file is an image file (e.g png, jpg, jpeg, gif, bmp) or another file. The problem is that I'm using Uploadify to upload the files, which changes the mime type and gives a 'text/octal' or something as the mime type, no matter which file type you upload.</p>"""

#print(tokenize(text))
#print(pos_tokens(text))
#print(freq_pos(text).most_common(100))
print(filtered_tokens(text))