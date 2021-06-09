from datetime import datetime
import pandas as pd

def Update_Data():
    #####Change file path#####
    Data_Base = pd.read_csv('~/DriveX/EMS_NRLab/Forecast_Cons/Last_day.csv', index_col=0)
    New_data1 = pd.read_csv('~/DriveX/EMS_NRLab/NRLab_state.csv', index_col=0)
    New_data2 = pd.read_csv('~/DriveX/EMS_NRLab/NRLab_meteo.csv', index_col=0)

    consumption = -New_data1["Power (W)"]["LOAD"]
    temp = New_data2["Value"]["Air_Temp"]
    humi = New_data2["Value"]["Relative_humidity"]
    ws = New_data2["Value"]["Wind Speed"]
    wd = New_data2["Value"]["Wind direction"]

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

    New = pd.DataFrame([{'Date and time (UTC)': dt_string, "TGBT":consumption,"AirTemp":temp, "wd":wd, 'ws':ws, "rh":humi}], index = [0]) 
    frames = [Data_Base,New]

    Data_Base = pd.concat(frames,ignore_index=True)

    if(len(Data_Base) > 5100):
        Data_Base = Data_Base.iloc[1:]

    #####Change file path#####
    Data_Base.to_csv('~/DriveX/EMS_NRLab/Forecast_Cons/Last_day.csv', index = True)

Update_Data()
