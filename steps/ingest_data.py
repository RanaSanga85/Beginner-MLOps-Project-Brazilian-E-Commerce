import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    It encapsulates the functionality of reading data from a given file path.
    """
    def __init__(self, data_path:str):
        self.data_path = data_path

    def get_data(self):
        """
        This method defines the functionality to read the data from the provided path.
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step   #This is a decorator provided by ZenML, which converts the function ingest_data into a ZenML pipeline step
def ingest_data(data_path: str):
    """
    This function is a step in the pipeline that takes data_path as an argument:
    """    
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()  #The get_data method of the IngestData instance is called, which 
                                     #reads the data from the file and returns it as a DataFrame
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e