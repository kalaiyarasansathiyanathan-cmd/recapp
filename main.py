import io
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import streamlit as st

st.set_page_config(
    page_title="ReCycle App",
    page_icon="♻️",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': '''
        <meta name="google-site-verification" content="YOUR_VERIFICATION_CODE_HERE" />
        '''
    }
)

app = FastAPI(title="Plastic Classification API")

# மொபைல் ஆப் தொடர்புகொள்ள CORS அனுமதி
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "plastic_classifier_model.keras"

plastic_full_names = {
    "PET": "PET (Polyethylene Terephthalate)",
    "HDPE": "HDPE (High-Density Polyethylene)",
    "PVC": "PVC (Polyvinyl Chloride)",
    "LDPE": "LDPE (Low-Density Polyethylene)",
    "PP": "PP (Polypropylene)",
    "PS": "PS (Polystyrene)"
}

class_names = ["HDPE", "LDPE", "PET", "PP", "PS", "PVC"]

recycling_ideas = {
    "English": {
        "PET": [
            {"text": "You can upcycle this bottle into a self-watering planter by cutting it in half and adding a cotton wick.", "icon": "https://img.icons8.com/color/96/potted-plant.png"},
            {"text": "It can be repurposed as a neat pencil holder for your study table or desk.", "icon": "https://img.icons8.com/color/96/pencil-holder.png"},
            {"text": "You can turn it into a simple piggy bank by making a small slot on top for saving coins.", "icon": "https://img.icons8.com/color/96/piggy-bank.png"}
        ],
        "HDPE": [
            {"text": "This sturdy container can be used to store household detergents, liquids, or cleaning supplies.", "icon": "https://img.icons8.com/color/96/cleaning-products.png"},
            {"text": "You can transform it into a durable watering can for your home garden by drilling small holes in the cap.", "icon": "https://img.icons8.com/color/96/watering-can.png"},
            {"text": "It can be cut and crafted into practical desk organizers for stationary items.", "icon": "https://img.icons8.com/color/96/desk.png"}
        ],
        "PVC": [
            {"text": "These pipes can be converted into custom desk organizers for tools and office accessories.", "icon": "https://img.icons8.com/color/96/toolbox.png"},
            {"text": "You can use them as flexible cable managers to neatly organize cluttered wires behind your desk.", "icon": "https://img.icons8.com/color/96/cable-release.png"},
            {"text": "They can be used as durable border edging for garden beds and outdoor pathways.", "icon": "https://img.icons8.com/color/96/fence.png"}
        ],
        "LDPE": [
            {"text": "You can weave clean plastic bags together into 'plarn' (plastic yarn) to make reusable shopping bags.", "icon": "https://img.icons8.com/color/96/shopping-bag.png"},
            {"text": "They can be fused together using an iron to create waterproof mats or protective covers.", "icon": "https://img.icons8.com/color/96/mat.png"}
        ],
        "PP": [
            {"text": "You can chop colorful caps into small tiles to create creative mosaic art projects.", "icon": "https://img.icons8.com/color/96/art.png"},
            {"text": "These containers can be cut and fitted as custom drawer dividers for small household items.", "icon": "https://img.icons8.com/color/96/drawer.png"},
            {"text": "They make excellent small seed starter pots for growing new plants at home.", "icon": "https://img.icons8.com/color/96/sprout.png"}
        ],
        "PS": [
            {"text": "You can craft foam or rigid polystyrene pieces into lightweight picture frames.", "icon": "https://img.icons8.com/color/96/picture-frame.png"},
            {"text": "It can be reused to build durable architectural or hobby models for DIY projects.", "icon": "https://img.icons8.com/color/96/craft.png"}
        ]
    },
    "Tamil": {
        "PET": [
            {"text": "இந்த பாட்டிலை பாதியாக வெட்டி, பஞ்சு திரி சேர்த்து தானாக நீர் பாய்ச்சும் செடி தொட்டியாக மாற்றலாம்.", "icon": "https://img.icons8.com/color/96/potted-plant.png"},
            {"text": "உங்கள் படிப்பு மேஜைக்கு அழகான பென்சில் ஹோல்டராக பயன்படுத்தலாம்.", "icon": "https://img.icons8.com/color/96/pencil-holder.png"},
            {"text": "மேல் பகுதியில் சிறிய துளையிட்டு நாணயங்களை சேமிக்கும் உண்டியலாக மாற்றலாம்.", "icon": "https://img.icons8.com/color/96/piggy-bank.png"}
        ],
        "HDPE": [
            {"text": "இந்த உறுதியான பாட்டிலை துப்புரவு திரவங்கள் அல்லது சோப்பு திரவங்களை சேமிக்க பயன்படுத்தலாம்.", "icon": "https://img.icons8.com/color/96/cleaning-products.png"},
            {"text": "மூடியில் சிறிய துளைகள் போட்டு தோட்டத்திற்கு நீர் பாய்ச்சும் பூவாளியாக மாற்றலாம்.", "icon": "https://img.icons8.com/color/96/watering-can.png"},
            {"text": "பொருட்களை அடுக்க வைக்கும் மேஜை அமைப்பாக வெட்டி பயன்படுத்தலாம்.", "icon": "https://img.icons8.com/color/96/desk.png"}
        ],
        "PVC": [
            {"text": "இந்த பைப்புகளை அலுவலக பொருட்கள் மற்றும் கருவிகளை வைக்கும் அமைப்பாக மாற்றலாம்.", "icon": "https://img.icons8.com/color/96/toolbox.png"},
            {"text": "மேஜை பின்னால் உள்ள ஒயர்களை ஒழுங்கமைக்க கேபிள் மேனேஜராக பயன்படுத்தலாம்.", "icon": "https://img.icons8.com/color/96/cable-release.png"},
            {"text": "தோட்டத்து பாதைகளுக்கு உறுதியான தடுப்பு வேலியாக பயன்படுத்தலாம்.", "icon": "https://img.icons8.com/color/96/fence.png"}
        ],
        "LDPE": [
            {"text": "பிளாஸ்டிக் பைகளை நெய்து மீண்டும் பயன்படுத்தக்கூடிய ஷாப்பிங் பைகளாக மாற்றலாம்.", "icon": "https://img.icons8.com/color/96/shopping-bag.png"},
            {"text": "அயர்ன் பாக்ஸ் மூலம் சூடுபடுத்தி நீர்ப்புகா பாய்களாக தயாரிக்கலாம்.", "icon": "https://img.icons8.com/color/96/mat.png"}
        ],
        "PP": [
            {"text": "வண்ண மூடிகளை சிறு துண்டுகளாக வெட்டி கலைப் பொருட்களாக மாற்றலாம்.", "icon": "https://img.icons8.com/color/96/art.png"},
            {"text": "டிராயர்களில் சிறிய பொருட்களை தனித்தனியாக பிரிக்க பயன்படுத்தலாம்.", "icon": "https://img.icons8.com/color/96/drawer.png"},
            {"text": "வீட்டில் புதிய செடிகளை வளர்க்க விதை தொட்டியாக பயன்படுத்தலாம்.", "icon": "https://img.icons8.com/color/96/sprout.png"}
        ],
        "PS": [
            {"text": "தெர்மோகோல் துண்டுகளை லேசான போட்டோ பிரேம்களாக மாற்றலாம்.", "icon": "https://img.icons8.com/color/96/picture-frame.png"},
            {"text": "கட்டிட மாதிரிகள் அல்லது கைவினை திட்டங்களுக்கு பயன்படுத்தலாம்.", "icon": "https://img.icons8.com/color/96/craft.png"}
        ]
    },
    "Malayalam": {
        "PET": [
            {"text": "ഈ കുപ്പി പകുതിയായി മുറിച്ച് പഞ്ഞി തിരിയിട്ട് സ്വയം വെള്ളമൊഴിക്കുന്ന ചെടിത്തൊട്ടിയാക്കാം.", "icon": "https://img.icons8.com/color/96/potted-plant.png"},
            {"text": "പഠനമേശയിൽ പെൻസിലുകൾ വെക്കാനുള്ള പാത്രമായി ഉപയോഗിക്കാം.", "icon": "https://img.icons8.com/color/96/pencil-holder.png"},
            {"text": "മുകളിൽ ചെറിയ ദ്വാരമിട്ട് നാണയങ്ങൾ സൂക്ഷിക്കുന്ന സിപ്പിയാക്കാം.", "icon": "https://img.icons8.com/color/96/piggy-bank.png"}
        ],
        "HDPE": [
            {"text": "സോപ്പ് ലായനികളും ക്ലീനിംഗ് സാമഗ്രികളും സൂക്ഷിക്കാൻ ഇത് ഉപയോഗിക്കാം.", "icon": "https://img.icons8.com/color/96/cleaning-products.png"},
            {"text": "അടപ്പിൽ ചെറിയ ദ്വാരങ്ങളിട്ട് ചെടികൾക്ക് വെള്ളമൊഴിക്കുന്ന പാത്രമാക്കാം.", "icon": "https://img.icons8.com/color/96/watering-can.png"},
            {"text": "മേശപ്പുറത്തെ സാധനങ്ങൾ ഒതുക്കി വെക്കാനുള്ള ബോക്സുകളാക്കാം.", "icon": "https://img.icons8.com/color/96/desk.png"}
        ],
        "PVC": [
            {"text": "ടൂളുകളും ഓഫീസിലെ സാധനങ്ങളും വെക്കാനുള്ള ഹോൾഡറുകളാക്കാം.", "icon": "https://img.icons8.com/color/96/toolbox.png"},
            {"text": "കേബിളുകൾ ഭംഗിയായി ഒതുക്കി വെക്കാൻ ഉപയോഗിക്കാം.", "icon": "https://img.icons8.com/color/96/cable-release.png"},
            {"text": "തോട്ട അതിരുകൾ നിർമ്മിക്കാൻ ഉപയോഗിക്കാം.", "icon": "https://img.icons8.com/color/96/fence.png"}
        ],
        "LDPE": [
            {"text": "പ്ലാസ്റ്റിക് കവറുകൾ നെയ്ത് വീണ്ടും ഉപയോഗിക്കാവുന്ന ബാഗുകളാക്കാം.", "icon": "https://img.icons8.com/color/96/shopping-bag.png"},
            {"text": "ഐൺ ബോക്സ് ഉപയോഗിച്ച് ചൂടാക്കി വാട്ടർപ്രൂഫ് മാറ്റുകളാക്കാം.", "icon": "https://img.icons8.com/color/96/mat.png"}
        ],
        "PP": [
            {"text": "കളർ അടപ്പുകൾ ചെറിയ കഷ്ണങ്ങളാക്കി ആർട്ട് പ്രൊജക്റ്റുകൾ ചെയ്യാം.", "icon": "https://img.icons8.com/color/96/art.png"},
            {"text": "ഡ്രോയറുകളിൽ സാധനങ്ങൾ തരംതിരിച്ചു വെക്കാൻ ഉപയോഗിക്കാം.", "icon": "https://img.icons8.com/color/96/drawer.png"},
            {"text": "ചെടികൾ മുളപ്പിച്ചെടുക്കാൻ ചെറിയ തൊട്ടികളായി ഉപയോഗിക്കാം.", "icon": "https://img.icons8.com/color/96/sprout.png"}
        ],
        "PS": [
            {"text": "തെർമോക്കോൾ കഷ്ണങ്ങൾ ഫോട്ടോ ഫ്രെയിമുകളാക്കി മാറ്റാം.", "icon": "https://img.icons8.com/color/96/picture-frame.png"},
            {"text": "മോഡലുകളും ക്രാഫ്റ്റ് വസ്തുക്കളും ഉണ്ടാക്കാൻ ഉപയോഗിക്കാം.", "icon": "https://img.icons8.com/color/96/craft.png"}
        ]
    },
    "Hindi": {
        "PET": [
            {"text": "इस बोतल को आधा काटकर और कॉटन की बत्ती लगाकर सेल्फ-वॉटरिंग पॉट बनाएं।", "icon": "https://img.icons8.com/color/96/potted-plant.png"},
            {"text": "इसे अपनी स्टडी टेबल के लिए पेंसिल होल्डर के रूप में इस्तेमाल करें।", "icon": "https://img.icons8.com/color/96/pencil-holder.png"},
            {"text": "ऊपर एक छोटा छेद करके सिक्के बचाने के लिए गुल्लक बनाएं।", "icon": "https://img.icons8.com/color/96/piggy-bank.png"}
        ],
        "HDPE": [
            {"text": "इस मजबूत डिब्बे का उपयोग डिटर्जेंट या सफाई के सामान को रखने के लिए करें।", "icon": "https://img.icons8.com/color/96/cleaning-products.png"},
            {"text": "ढक्कन में छोटे छेद करके अपने बगीचे के लिए वॉटरिंग कैन बनाएं।", "icon": "https://img.icons8.com/color/96/watering-can.png"},
            {"text": "इसे काटकर टेबल ऑर्गनाइज़र के रूप में उपयोग करें।", "icon": "https://img.icons8.com/color/96/desk.png"}
        ],
        "PVC": [
            {"text": "इन पाइपों को औजारों और ऑफिस सामान के ऑर्गनाइज़र में बदलें।", "icon": "https://img.icons8.com/color/96/toolbox.png"},
            {"text": "टेबल के पीछे बिखरे तारों को व्यवस्थित करने के लिए उपयोग करें।", "icon": "https://img.icons8.com/color/96/cable-release.png"},
            {"text": "बगीचे की क्यारियों के लिए मजबूत बॉर्डर बनाएं।", "icon": "https://img.icons8.com/color/96/fence.png"}
        ],
        "LDPE": [
            {"text": "प्लास्टिक बैग्स को बुनकर दोबारा इस्तेमाल योग्य शॉपिंग बैग बनाएं।", "icon": "https://img.icons8.com/color/96/shopping-bag.png"},
            {"text": "प्रेस (Iron) से जोड़कर वॉटरप्रूफ मैट बनाएं।", "icon": "https://img.icons8.com/color/96/mat.png"}
        ],
        "PP": [
            {"text": "रंगीन ढक्कनों को छोटे टुकड़ों में काटकर आर्ट प्रोजेक्ट बनाएं।", "icon": "https://img.icons8.com/color/96/art.png"},
            {"text": "दराजों (Drawers) में छोटे सामान को अलग रखने के लिए उपयोग करें।", "icon": "https://img.icons8.com/color/96/drawer.png"},
            {"text": "पौधे उगाने के लिए छोटे गमलों के रूप में उपयोग करें।", "icon": "https://img.icons8.com/color/96/sprout.png"}
        ],
        "PS": [
            {"text": "थर्माकोल के टुकड़ों से हल्के फोटो फ्रेम बनाएं।", "icon": "https://img.icons8.com/color/96/picture-frame.png"},
            {"text": "डीआईवाई प्रोजेक्ट्स और क्राफ्ट मॉडल बनाने में उपयोग करें।", "icon": "https://img.icons8.com/color/96/craft.png"}
        ]
    },
    "Telugu": {
        "PET": [
            {"text": "ఈ సీసాని సగానికి కోసి, కాటన్ వత్తిని అమర్చి సెల్ఫ్-వాటరింగ్ ప్లాంటర్‌గా మార్చవచ్చు.", "icon": "https://img.icons8.com/color/96/potted-plant.png"},
            {"text": "మీ స్టడీ టేబుల్ కోసం పెన్సిల్ హోల్డర్‌గా ఉపయోగించవచ్చు.", "icon": "https://img.icons8.com/color/96/pencil-holder.png"},
            {"text": "నాణేలు దాచుకోవడానికి చిన్న రంధ్రం చేసి పిగ్గీ బ్యాంక్‌గా మార్చవచ్చు.", "icon": "https://img.icons8.com/color/96/piggy-bank.png"}
        ],
        "HDPE": [
            {"text": "ఈ గట్టి బాటిల్‌ను డిటర్జెంట్లు లేదా లిక్విడ్‌లను దాచడానికి ఉపయోగించవచ్చు.", "icon": "https://img.icons8.com/color/96/cleaning-products.png"},
            {"text": "మూతకి చిన్న రంధ్రాలు చేసి మొక్కలకు నీళ్లు పోసే కాన్‌గా మార్చవచ్చు.", "icon": "https://img.icons8.com/color/96/watering-can.png"},
            {"text": "టేబుల్ ఆర్గనైజర్‌గా కట్ చేసి ఉపయోగించవచ్చు.", "icon": "https://img.icons8.com/color/96/desk.png"}
        ],
        "PVC": [
            {"text": "ఈ పైపులను ఆఫీస్ టూల్స్ మరియు వైర్లు అమర్చుకోవడానికి ఉపయోగించవచ్చు.", "icon": "https://img.icons8.com/color/96/toolbox.png"},
            {"text": "కేబుల్స్ సరిగ్గా అమర్చడానికి కేబుல் మేనేజర్‌గా ఉపయోగించవచ్చు.", "icon": "https://img.icons8.com/color/96/cable-release.png"},
            {"text": "గార్డెన్ సరిహద్దుల కోసం ఉపయోగించవచ్చు.", "icon": "https://img.icons8.com/color/96/fence.png"}
        ],
        "LDPE": [
            {"text": "ప్లాస్టిక్ కవర్లను అల్లి షాపింగ్ బ్యాగులుగా తయారు చేయవచ్చు.", "icon": "https://img.icons8.com/color/96/shopping-bag.png"},
            {"text": "ఐరన్ బాక్స్‌తో వేడి చేసి వాటర్‌ప్రൂఫ్ మ్యాట్‌లుగా మార్చవచ్చు.", "icon": "https://img.icons8.com/color/96/mat.png"}
        ],
        "PP": [
            {"text": "రంగు మూతలను చిన్న ముక్కలుగా చేసి ఆర్ట్ ప్రాజెక్ట్‌లు చేయవచ్చు.", "icon": "https://img.icons8.com/color/96/art.png"},
            {"text": "డ్రాయర్లలో చిన్న వస్తువులను సర్దుకోవడానికి ఉపయోగించవచ్చు.", "icon": "https://img.icons8.com/color/96/drawer.png"},
            {"text": "చిన్న మొక్కలు పెంచడానికి కుండీలుగా ఉపయోగించవచ్చు.", "icon": "https://img.icons8.com/color/96/sprout.png"}
        ],
        "PS": [
            {"text": "థర్మోకోల్ ముక్కలతో తేలికపాటి ఫోటో ఫ్రేమ్‌లు చేయవచ్చు.", "icon": "https://img.icons8.com/color/96/picture-frame.png"},
            {"text": "క్రాఫ్ట్ మోడల్స్ తయారు చేయడానికి ఉపయోగించవచ్చు.", "icon": "https://img.icons8.com/color/96/craft.png"}
        ]
    },
    "Kannada": {
        "PET": [
            {"text": "ಈ సీಸೆಯನ್ನು ಅರ್ಧಕ್ಕೆ ಕತ್ತರಿಸಿ, ಹತ್ತಿ ಬತ್ತಿ ಹಾಕಿ ಸ್ವಯಂ-ನೀರೆರೆಯುವ ಕುಂಡವಾಗಿ ಮಾಡಬಹುದು.", "icon": "https://img.icons8.com/color/96/potted-plant.png"},
            {"text": "ನಿಮ್ಮ ಅಧ್ಯಯನ ಮೇಜಿಗೆ ಪೆನ್ಸಿಲ್ ಹೋಲ್ಡರ್ ಆಗಿ ಬಳಸಿ.", "icon": "https://img.icons8.com/color/96/pencil-holder.png"},
            {"text": "ಮೇಲ್ಭಾಗದಲ್ಲಿ ಸಣ್ಣ ರಂಧ್ರ ಮಾಡಿ ನಾಣ್ಯ ಉಳಿಸುವ ಗಿಲ್ಲಿ ಮಾಡಬಹುದು.", "icon": "https://img.icons8.com/color/96/piggy-bank.png"}
        ],
        "HDPE": [
            {"text": "ಈ ಗಟ್ಟಿ ಬಾಟಲಿಯನ್ನು ಡಿಟರ್ಜೆಂಟ್ ಅಥವಾ ಕ್ಲೀನಿಂಗ್ ದ್ರವಗಳನ್ನು ಇಡಲು ಬಳಸಿ.", "icon": "https://img.icons8.com/color/96/cleaning-products.png"},
            {"text": "ಮೂಡಿಗೆ ಸಣ್ಣ ರಂಧ್ರ ಮಾಡಿ ಗಿಡಗಳಿಗೆ ನೀರು ಹಾಕುವ ಕ್ಯಾನ್ ಮಾಡಬಹುದು.", "icon": "https://img.icons8.com/color/96/watering-can.png"},
            {"text": "ಡೆಸ್ಕ್ ಆರ್ಗನೈಸರ್ ಆಗಿ ಕತ್ತರಿಸಿ ಬಳಸಬಹುದು.", "icon": "https://img.icons8.com/color/96/desk.png"}
        ],
        "PVC": [
            {"text": "ಈ ಪೈಪ್‌ಗಳನ್ನು ಆಫೀಸ್ ಉಪಕರಣಗಳನ್ನು ಇಡುವ ಹೋಲ್ಡರ್ ಆಗಿ ಮಾರ್ಪಡಿಸಿ.", "icon": "https://img.icons8.com/color/96/toolbox.png"},
            {"text": "ವೈರ್‌ಗಳನ್ನು ನೀಟಾಗಿ ಜೋಡಿಸಲು ಕೇಬಲ್ ಮೇನೇಜರ್ ಆಗಿ ಬಳಸಿ.", "icon": "https://img.icons8.com/color/96/cable-release.png"},
            {"text": "ತೋಟದ ಗಡಿಗಳಿಗೆ ತಡೆಯಾಗಿ ಬಳಸಬಹುದು.", "icon": "https://img.icons8.com/color/96/fence.png"}
        ],
        "LDPE": [
            {"text": "ಪ್ಲಾಸ್ಟಿಕ್ ಚೀಲಗಳನ್ನು ಹೆಣೆದು ಮರುಬಳಕೆಯ ಶೋಪಿಂಗ್ ಬ್ಯಾಗ್ ಮಾಡಬಹುದು.", "icon": "https://img.icons8.com/color/96/shopping-bag.png"},
            {"text": "ಐರನ್ ಬಾಕ್ಸ್ ಬಳಸಿ ಬಿಸಿ ಮಾಡಿ ವಾಟರ್‌ಪ್ರೂಫ್ ಮ್ಯಾಟ್‌ಗಳನ್ನು ಮಾಡಬಹುದು.", "icon": "https://img.icons8.com/color/96/mat.png"}
        ],
        "PP": [
            {"text": "ಬಣ್ಣದ ಮುಚ್ಚಳಗಳನ್ನು ಸಣ್ಣ ತುಂಡುಗಳಾಗಿ ಕತ್ತರಿಸಿ ಆರ್ಟ್ ಪ್ರಾಜೆಕ್ಟ್ ಮಾಡಬಹುದು.", "icon": "https://img.icons8.com/color/96/art.png"},
            {"text": "ಡ್ರಾಯರ್‌ಗಳಲ್ಲಿ ವಸ್ತುಗಳನ್ನು ಪ್ರತ್ಯೇಕಿಸಲು ಬಳಸಿ.", "icon": "https://img.icons8.com/color/96/drawer.png"},
            {"text": "ಸಣ್ಣ ಗಿಡಗಳನ್ನು ಬೆಳೆಸಲು ಕುಂಡಗಳಾಗಿ ಬಳಸಿ.", "icon": "https://img.icons8.com/color/96/sprout.png"}
        ],
        "PS": [
            {"text": "ಥರ್ಮೋಕೋಲ್ ತುಂಡುಗಳಿಂದ ಹಗುರವಾದ ಫೋಟೋ ಫ್ರೇಮ್‌ಗಳನ್ನು ಮಾಡಬಹುದು.", "icon": "https://img.icons8.com/color/96/picture-frame.png"},
            {"text": "ಕ್ರಾಫ್ಟ್ ಮಾಡೆಲ್‌ಗಳನ್ನು ಮಾಡಲು ಬಳಸಿ.", "icon": "https://img.icons8.com/color/96/craft.png"}
        ]
    }
}

# Keras Model Load
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = None

@app.get("/")
def home():
    return {"message": "Plastic Classification API is Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), language: str = Form("English")):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not loaded")
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_idx = int(np.argmax(predictions[0]))
        predicted_short = class_names[predicted_idx]
        predicted_full = plastic_full_names.get(predicted_short, predicted_short)
        confidence = round(float(np.max(predictions[0])) * 100, 2)

        lang_ideas = recycling_ideas.get(language, recycling_ideas["English"])
        suggestions = lang_ideas.get(predicted_short, [])

        return {
            "predicted_type": predicted_full,
            "short_code": predicted_short,
            "confidence": confidence,
            "language": language,
            "suggestions": suggestions
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
