from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

sent = """all_exp_traintest &lt;- readMat(all_exp_filepath);
len = length(all_exp_traintest$exp.traintest)/2;
    for (i in 1:len) {
      expert_train_df &lt;- data.frame(all_exp_traintest$exp.traintest[i]);
      labels = data.frame(all_exp_traintest$exp.traintest[i+302]);
      names(labels)[1] &lt;- ""t_labels"";
      expert_train_df$t_labels &lt;- labels;
      expert_data_frame &lt;- data.frame(expert_train_df);
      rf_model = randomForest(expert_data_frame$t_labels ~., data=expert_data_frame, importance=TRUE, do.trace=100);
    }"""

sent2 = "c# c++  c a b k for int i=0;i<n;i++ build onresume upload file-upload php file xsp4"

print(pos_tag(word_tokenize(sent2)))

#sent3 = """"intel-rst": 41894, "httppostedfilebase": 32628, "datatemplates": 41896, "splines": 41897, "linkpoint": 42015, "git-submodules": 41899, "public-domain": 41900, "android-event": 41901, "dynamic-cast": 41902, "socket-server": 41903, "iphone-private-api": 41904, "interactive-login": 41905, "burst-mode": 41906, "68000": 38392, "java-deployment-toolkit": 41908, "sequence": 41308, "settext": 41910, "searching": 41911, "jscalendar": 41912, "110": 41913, "django-uploads": 41914, "sailfish-os": 41915, "cfthread": 41916, "lead": 41917, "cocoalibspotify-2.0": 41918, "leak": 41919, "dynamically-generated": 31745, "cross-domain": 41921, "domain-registrar": 41922, "page-caching": 41923, "table-per-hierachy": 41924, "artifactory": 13612, "locate": 41926, "11g": 41927, "fastinfoset": 41928, "enum": 41929, "character-replacement": 41930, "mitk": 41931, "app-tabs": 41932, "nscountedset": 21956, "sysinfo": 41934, "slug": 41935, "statspack": 4069, "intel-celeron": 41937, "shipping": 41497, "strassen": 21767, "nscache": 41940, "winddk": 41941, "oracle-adf": 41942, "exposure-correction": 41521, "pynotify": 41944, "development-fabric": 42027, "warranty": 41946, "uniform-integrability": 41947, "pointer-arithmetic": 41948, "apache-fop": 41949, "libtiff": 41950, "getschematable": 41951, "umbraco": 41952, "scrummaster": 41953, "brush": 41954, "registration": 28093, "page-zoom": 41956, "indefinite-integral": 41957, "manageddirectx": 41958, "editing-menu": 36918, "memory-access": 41959, "secd-machine": 41960, "wii-u": 41961, "opennebula": 41962, "network-map": 41963, "google-data-api": 41964, "mozy": 41965, "physx": 41966, "video-camera": 41967, "log4net": 19933, "gwt-2.3": 41969, "multigrid": 41970, "gwt-2.4": 41971, "gwt-2.5": 41972, "text-driver": 41973, "libmagic": 41974, "compaq-visual-fortran": 41975, "blobstorage": 41976, "mobile-controls": 41977, "3d-reconstruction": 41978, "ixmldomelement": 41979, "regexkitlite": 41715, "mashery": 41981, "siteedit": 41982, "lvs": 41983, "projective-geometry": 41984, "ghost4j": 41985, "icollection": 20341, "default-implementation": 41987, "unary-languages": 41988, "tactionmanager": 41989, "lvm": 41990, "backport": 34670, "gaps-and-islands": 41992, "metric-system": 41993, "php-5.2": 41994, "ppmcolormask": 41995, "install-name-tool": 41996, "prerender": 27729, "three-tier": 41998, "partially-applied-type": 32789, "dotnet-framework": 42000, "robospice": 42001, "context-sensitive-help": 42002, "jmdns": 9186, "configurationsection": 42004, "freelancing": 42005, "sessioncontext": 42006, "xsbt-web-plugin": 42007, "percent": 41857, "sicp": 41859, "peoplesoft": 42010, "mcapi": 41246, "book": 41876, "branch": 42013, "theos": 42014, "memoize": 41898, "console-redirect": 42016, "perl-xs": 42017, "table-partitioning": 42018, "interop-domino": 42019, "mobile-application": 42020, "snapstodevicepixels": 42021, "week-number": 42022, "jung": 42023, "xmldataset": 42024, "inappstorewindow": 42025, "lens-construction": 42026, "asset-pipeline": 41945, "fiddler2": 42028, "contiki": 17580, "ifilter": 4762, "predicate-logic": 42031, "autoversioning": 42032, "tracekit": 42033, "jscript": 42034, "space": 2882, "geometric-construction": 16442, "delphi-3": 42037, "setupapi": 42038, "jstorage": 42039, "moderncv": 37343, "xsp4": 42041, "mac-office": 42042, "gkmatchmaker": 42044, "vim-plugins": 42045, "numeric-conversion": 42046, "urchin": 42047, "uncle-bob": 42043"""
filepath = "../../Data/all_labels.json"

f = open(filepath,'r')
sent3 = "" + f.read()
l = (pos_tag(word_tokenize(sent2)))
count = 0
M = {}
for each in l:
    #print (each)
    #if each[1] not in ['NN', 'NNS', 'VBG','VBN','VBD','VBZ','VBP','RB','VB', 'JJ','JJS','JJR','#','VBD',')','CD','.',';',':',',','``',"''"]:
    #print(each)
    if M.get(each[1]) == None:
        M[each[1]] = 1
    else:
        M[each[1]] = M[each[1]]+1;

print(M)
f.close()

#for each in M:
#    print(each + "-" + str(M.get(each)))
