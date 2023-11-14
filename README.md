# yoga_app

## Getting Started
 - run
 
    python3.11 -m venv venv

 - run

    source venv/bin/activate

 - run 

    pip install -r requirements.txt


## Folder structure 

 - Create a folder assets in the root directory
 - Create a folder images in the assets directory
 - download the images from the link below and save them in the images folder :
 https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset

 Then copy paste the test/train in images folder


 WARNING : Everything folder should be name in lowerCase


 Expected structure :

 YOGA_APP
│
├── assets
│   ├── config
│   └── images
│       ├── test
│       └── train
│
├── docs
│
├── PoseClassification
│   ├── __pycache__
│   ├── bootstrap.py
│   ├── khouluevre 2.py
│   ├── khouluevre.py
│   ├── pose_classifier.py
│   ├── pose_embedding.py
│   ├── utils.py
│   └── visualize.py
│
├── venv
│   ├── bin
│   ├── include
│   ├── lib
│   └── pyvenv.cfg
│
├── .gitignore
├── LICENSE
├── main.ipynb
├── README.md
└── tuto.ipynb
