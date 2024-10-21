# CALCU ML API

## Getting Started with Setting up Environment
```
git clone git@github.com:namespace-team/calcu_ml-api.git
cd calcu_ml-api
pip install -r requirements.txt
```
In case it requires to install additional python packages, run `pip install package_name` or `pip3 install package_name`

NOTE: Requires `Anaconda 3` and `python 3.7 or higher`. Please check last section for installation documentation

## Authorization and model loading
1. Create a file named `storage_credentials.json` in project root directory
2. Copy credentials content from [here](https://namespace-inc.atlassian.net/wiki/spaces/NI/pages/2057863171/CALCU+-+ML+-+Contribution#Credentials) and paste to `storage_credentials.json`
3. Run `sh download_weights.sh` in terminal to create and download yolo model and labels

## Making the prediction

### Run the server
NOTE: To run the server make sure you have correct `conda env` with correct `python version`.
Please check how to create and activate correct conda env below(Required to install Anaconda first)
```
uvicorn app.main:app --host 0.0.0.0 --port 8080 (Specify different port if 8080 is in use by other app)
uvicorn app.main:app --reload (--reload option reloads the response without no need to stop and start the server for new response)
```

### Getting ML response
1. Open postman desktop agent app
2. Open a new tab
3. Paste this url in url section `http://0.0.0.0:8080/download_and_predict`
4. Select POST method
5. Select `raw` and type as `JSON` in body section
6. Send the request for before_link and after_link in body section. Refer example as below:
```
{
    "before_link": "https://firebasestorage.googleapis.com/v0/b/calcu-staging.appspot.com/o/upload%2FThu%20May%2026%2014%3A56%3A00%202022%2Fmrousavy889094741906192538.jpg?alt=media&token=1c3eac5a-80c0-4b2e-abe0-5b6c685c0f66",
    "after_link": "https://firebasestorage.googleapis.com/v0/b/calcu-staging.appspot.com/o/upload%2FThu%20May%2026%2014%3A56%3A54%202022%2Fmrousavy4074231933553723946.jpg?alt=media&token=119021a3-8977-49f5-bd26-cdbd44d83846",
    "return_img": "true"
}
```
7. Click `Send`

## Installing Anaconda 3

1. Follow this [link](https://docs.anaconda.com/anaconda/install/linux/) to install Anaconda 3(for Python 3.7)


## Creating and activating conda environment
1. Run `conda env list` to check list of conda environments
2. Run `conda create --name calcu-ml-api-3.7 python=3.7.0` to create a new conda env namely `calcu-ml-api-3.7` with python version `3.7.0`
3. Similarly, Run `conda create --name calcu-ml-api python=3.9.5` to create a new conda env namely `calcu-ml-api` with python version `3.9.5`
4. Run `conda activate calcu-ml-api-3.7` to activate `calcu-ml-api-3.7` env
