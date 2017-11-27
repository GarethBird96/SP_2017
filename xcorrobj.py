import numpy as np
import array 
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
class crosscorrelator():
    def __init__(self):
        #Intialise List of arrays
	self.datastore=list()
	self.timestamps=list()
    def add_correlation_data_channel(self,channel_number,data_list):
        #if list entry empty, create array
        print type(channel_number)
	for time_index in range(len(data_list)):
            if len(self.datastore)<= time_index:
                self.datastore.append(array.array('f'))
            self.datastore[time_index].insert(channel_number,data_list[time_index])
    def add_all_signals(self,ms_output_data,sat_num,signal_type):
        #load float infos
        signaldata = ms_output_data[sat_num][signal_type]
        number_of_channels = len(signaldata[0])
        for channel in range(number_of_channels):
            chsignal = [signal[channel] for signal in signaldata ]
            self.add_correlation_data_channel(channel,chsignal)
    def add_time_data(self,timestamps):
        self.timestamps=timestamps
    def correlate(self,corrinterval,ch1,ch2):
	#Check if data makes sense
	if len(self.datastore) != len(self.timestamps):
            raise ValueError ("Mismatched Data Sizes")
        if type(corrinterval) != type(timedelta()):
            raise ValueError ("Not of timedelta format ")

        new_correlation= True;
        time_index = int(0)
        starting_index = int(0)
        correlation_data=list()
        self.interval_datetimes = list()
        #Run across datetimes
        for datetime in self.timestamps:
            #reset vals if new time interval
            if(new_correlation):
                timesegment = timedelta(0)
                ch1data=list()
                ch2data=list()
                new_correlation = False
                start_time = datetime;
                self.interval_datetimes.append(start_time)

            #append array until time exceed timewidth
            ch1data.append(self.datastore[time_index][ch1])
            ch2data.append(self.datastore[time_index][ch2])
            timesegment= datetime - start_time
            if(corrinterval<timesegment):
                #Once over time interval: calc correlation for the interval and append to the list to return
                new_correlation = True
                corval=self.calculate_correlation(np.asarray(ch1data),np.asarray(ch2data))
                correlation_data.append(corval)
                #create vectors to calculate xcorr 
            time_index = time_index + 1
	return correlation_data
    def calculate_correlation(self,array1,array2):
        return np.correlate(array1, array2)/(math.sqrt(np.sum(array1**2)*np.sum(array2**2)))
    def create_corr_matrix(self,timewidth):
        channels=len(self.datastore[0])
        conversion_step=list()
        for ch1 in range(len(self.datastore[0])):
            for ch2 in range(len(self.datastore[0])):
                 if ch1>=ch2:
                    convert=np.array(self.correlate(timewidth,ch1,ch2)).tolist()
                    #Create tuple of form (ch1,ch2,np.array of data
                    conversion_step.append((ch1,ch2,[convert[index][0] for index in range(len(convert))]))
        self.correlation_matrix=np.zeros((channels,channels,len(conversion_step[0][2])))
        for correlation_tuple in conversion_step :
            index = int(0)
            for corr_vals in correlation_tuple[2]:
                self.correlation_matrix[correlation_tuple[0],correlation_tuple[1],index]=corr_vals
#                print correlation_tuple[0],correlation_tuple[1], corr_vals, index
#                print self.correlation_matrix[correlation_tuple[0],correlation_tuple[1],index]
                index = index + 1
    def create_NK_plot(self,ch1,ch2,nkdata):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        nk_corr = nkdata[ch1,ch2]
        bg_corr = self.correlation_matrix[ch1,ch2]
        ax.hist(bg_corr, bins = np.arange(0, 1.01, 0.05), label = 'No tests')
        ax.hist(nk_corr, np.arange(0, 1.01, 0.05),
        color = 'red', label = 'Test: 25th May 2009')
        #plt.title(fname)
        plt.xlabel('Cross-correlation coefficient', fontsize = 30)
        plt.ylabel('Frequency', fontsize = 30)
        plt.legend(fontsize = 'xx-large')
        plt.title("ch1,ch2: "+str(ch1)+","+str(ch2))
        plt.show()
                        
