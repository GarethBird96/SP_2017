import numpy as np
import array 
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
class crosscorrelator():
    #Constructor
    def __init__(self):
        #Intialise List of arrays
	self.__datastore=list()
	self.__timestamps=list()

    def add_correlation_data_channel(self,channel_number,data_list):
        #if list entry empty, create array
	for time_index in range(len(data_list)):
            if len(self.__datastore)<= time_index:
                self.__datastore.append(array.array('f'))
            self.__datastore[time_index].insert(channel_number,data_list[time_index])

    #Appends data signals given signal info into class
    def add_all_signals(self,ms_output_data,sat_num,signal_type):
        #load float infos
        signaldata = ms_output_data[sat_num][signal_type]
        number_of_channels = len(signaldata[0])
        for channel in range(number_of_channels):
            chsignal = [signal[channel] for signal in signaldata ]
            self.add_correlation_data_channel(channel,chsignal)
        #appends bad data bool as a member
        self.__badindex = [ bool(vals==1) for vals in ms_output_data[sat_num]['dropped_data'] ]

    #Adds time data
    def add_time_data(self,timestamps):
        self.__timestamps=timestamps

    #Create correlation dataset for a pair of channels
    #Execute via create matrix methods
    def __correlate(self,corrinterval,ch1,ch2,normflag=True):
	#Check if data makes sense
	if len(self.__datastore) != len(self.__timestamps):
            raise ValueError ("Mismatched Data Sizes")
        if type(corrinterval) != type(timedelta()):
            raise ValueError ("Not of timedelta format ")
        #intialise data and booleans
        new_correlation= True;
        time_index = int(0)
        starting_index = int(0)
        correlation_data=list()
        self.interval_datetimes = list()
        self.interval_baddata = list()
        self.interval_startingindex = list()
        #Run across datetimes
        for datetime in self.__timestamps:
            #reset vals if new time interval
            if(new_correlation):
                timesegment = timedelta(0)
                ch1data=list()
                ch2data=list()
                new_correlation = False
                start_time = datetime;
                self.interval_datetimes.append(start_time)
                badflag = False
                self.interval_startingindex.append(time_index)
            #append array until time exceed timewidth
            ch1data.append(self.__datastore[time_index][ch1])
            ch2data.append(self.__datastore[time_index][ch2])
            timesegment= datetime - start_time
            #If value has baddata label , flag it
            if(self.__badindex[time_index]):
                badflag = True
            if(corrinterval<timesegment):
                #if badflag a
                #Once over time interval: calc correlation for the interval and append to the list to return
                self.interval_baddata.append(badflag)
                if normflag:
                    corval=self.__calculate_correlation(np.asarray(ch1data),np.asarray(ch2data))
                else:
                    corval=self.__calculate_nonorm_correlation(np.asarray(ch1data),np.asarray(ch2data))
                correlation_data.append(corval)
                new_correlation = True 
            time_index = time_index + 1
	return correlation_data

    #Numerical Calculation of Filip's Normalised Cross Correlation Values
    def __calculate_correlation(self,array1,array2):
        return np.correlate(array1, array2)/(math.sqrt(np.sum(array1**2)*np.sum(array2**2)))

    def create_corr_matrix(self,timewidth):
        self.timewidth=timewidth
        channels=len(self.__datastore[0])
        conversion_step=list()
        for ch1 in range(len(self.__datastore[0])):
            for ch2 in range(len(self.__datastore[0])):
                 if ch1>=ch2:
                    convert=np.array(self.__correlate(timewidth,ch1,ch2)).tolist()
                    #Create tuple of form (ch1,ch2,np.array of data) to be converted to numpy array
                    conversion_step.append((ch1,ch2,[convert[index][0] for index in range(len(convert))]))
        self.correlation_matrix=np.zeros((channels,channels,len(conversion_step[0][2])))
        for correlation_tuple in conversion_step :
            index = int(0)
            for corr_vals in correlation_tuple[2]:
                self.correlation_matrix[correlation_tuple[0],correlation_tuple[1],index]=corr_vals
                self.correlation_matrix[correlation_tuple[1],correlation_tuple[0],index]=corr_vals
                index = index + 1

    #Correlation Between two arrays for other analysis types
    def __calculate_nonorm_correlation(self,array1,array2):
        return np.correlate(array1, array2)

    def create_nonorm_corr_matrix(self,timewidth):
        channels=len(self.__datastore[0])
        conversion_step=list()
        for ch1 in range(len(self.__datastore[0])):
            for ch2 in range(len(self.__datastore[0])):
                 if ch1>=ch2:
                    convert=np.array(self.__correlate(timewidth,ch1,ch2,normflag=False)).tolist()
                    #Create tuple of form (ch1,ch2,np.array of data
                    conversion_step.append((ch1,ch2,[convert[index][0] for index in range(len(convert))]))
        self.nonorm_correlation_matrix=np.zeros((channels,channels,len(conversion_step[0][2])))
        for correlation_tuple in conversion_step :
            index = int(0)
            for corr_vals in correlation_tuple[2]:
                self.nonorm_correlation_matrix[correlation_tuple[0],correlation_tuple[1],index]=corr_vals
                self.nonorm_correlation_matrix[correlation_tuple[1],correlation_tuple[0],index]=corr_vals
                index = index + 1

    #Create Versions of Filip's Plot
    def create_NK_plot(self,ch1,ch2,nkdata,filename=''):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        nk_corr = nkdata[ch1,ch2]
        bg_corr = self.correlation_matrix[ch1,ch2]
        ax.hist(bg_corr, bins = np.arange(0, 1.01, 0.05), label = 'No tests')
        ax.hist(nk_corr, np.arange(0, 1.01, 0.05),
        color = 'red', label = 'Test: 25th May 2009')
        plt.xlabel('Cross-correlation coefficient', fontsize = 30)
        plt.ylabel('Frequency', fontsize = 30)
        plt.legend(fontsize = 'xx-large')
        plt.title("ch1,ch2: "+str(ch1)+","+str(ch2))
        if filename != '':
            plt.savefig(filename+str(ch1)+'_'+str(ch2)+'.svg')
        else:
            plt.show()
    #For current datasets + correlation interval, find p val of given
    def find_pval(self,x_corr_val,ch1,ch2):
        backgroundvals = self.correlation_matrix[ch1,ch2]
        return sum(i<x_corr_val for i in backgroundvals)/float(len(backgroundvals))

    #For given pair of channels return closest val data
    def find_closest_x_corr_index(self,x_corr_val,ch1,ch2):
        index , truevalue = self.__find_nearest(self.correlation_matrix[ch1,ch2],x_corr_val)
        time = self.interval_datetimes[index]
        return index , truevalue , time

    #for find x_corr_index, returns closest value
    def __find_nearest(self,array,value):
        idx = (np.abs(array-value)).argmin()
        return idx , array[idx]

    #Creates Scatters for a pair of channels of different crosscorrelation values
    def createscatters(self,min_val,max_val,ch1,ch2,shape=[3,2],filename='',colour='blue'):
        #Lineraly split values for plot
        plot_int_index = int(1)
        print shape
        fig = plt.figure(figsize=(5*shape[1],3*shape[0])) 
        for x_cor_val in np.linspace(min_val,max_val,num=(shape[0]*shape[1])):
            #Retrieve data
            index , truevalue , time = self.find_closest_x_corr_index(x_cor_val,ch1,ch2)
            #Find index in raw data
            rawstartindex = self.interval_startingindex[index]
            #Find data points from starting index info
            if (len(self.interval_startingindex)>index+1):
                rawendindex = self.interval_startingindex[index+1]
            else:
                rawendindex = len(self.interval_startingindex)-1
            x1 = [datapoints[ch1] for datapoints in self.__datastore[rawstartindex:rawendindex]]
            x2 = [datapoints[ch2] for datapoints in self.__datastore[rawstartindex:rawendindex]]
            ax = plt.subplot(shape[0],shape[1],plot_int_index)
            plt.scatter(x1,x2, color=colour)
            ax.set_xlabel('Signal from channel '+str(ch1))
            ax.set_ylabel('Signal from channel '+str(ch2))
            ax.set_title("ch"+str(ch1)+"ch"+str(ch2)+"\n X Corr Val:"+str(truevalue)+"\nBad Data:"+str(self.interval_baddata[index]))
            plot_int_index+=1
        plt.tight_layout()
        if filename != '':
            plt.savefig(filename+str(ch1)+'_'+str(ch2)+'.svg')
        else:
            plt.show()

    #Removes interval data that is considered off and returns adjused matrix, if replace_flag true the matrix is overwritten
    def remove_bad_intervals(self,replace_flag= False ):
        if (len(self.correlation_matrix[0,0]) != len(self.interval_baddata)):
            raise ValueError ("Mismatched Sizes of correlation values and bad interval list, likely cause is this command has already executed")
        badindices=list()
        for interval_index in range(len(self.correlation_matrix[0,0])):
            if (self.interval_baddata[interval_index]):
                badindices.append(interval_index)
        reduced_array = np.delete(self.correlation_matrix,badindices,axis=2)
        if(replace_flag):
            self.correlation_matrix= reduced_array
        print 'number of intervals removed ',len(badindices)
        return reduced_array

#Slightly modified copy of the procedure in the notebooks to generate and save all cross correlator plots for a given interval
def fulldataconstruction(satnum,output_data,nkoutput_data,correlation_interval):
    #create instances of cross-correlator
    background = crosscorrelator()
    nkevent = crosscorrelator()
    signal = 'rate_electron_measured'
    #Create Directory if it doesn't exsist
    dirpath='gareth/figures/ns'+str(satnum)+'_'+str(int((correlation_interval.total_seconds())/3600))+'hours'
    print dirpath
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    #Add data to clasa
    background.add_all_signals(output_data,satnum,signal)
    background.add_time_data(output_data[satnum]['datetime'][:])
    nkevent.add_all_signals(nkoutput_data,satnum,signal)
    nkevent.add_time_data(nkoutput_data[satnum]['datetime'][:])
    half_interval=timedelta(seconds=correlation_interval.total_seconds() * 0.5)
    background.create_corr_matrix(correlation_interval)
    nkevent.create_corr_matrix(correlation_interval)
    background.create_nonorm_corr_matrix(correlation_interval)
    nkevent.create_nonorm_corr_matrix(correlation_interval)
    min_xcorr_list= list()
    for ch1 in range(11):
        for ch2 in range(11):
            if ch2<ch1:
                savedir=dirpath+'/NKspectra'
                background.create_NK_plot(ch1,ch2,nkevent.correlation_matrix,savedir)
                minvalindex=np.argmin(nkevent.correlation_matrix[ch1,ch2])
                minval= nkevent.correlation_matrix[ch1,ch2,minvalindex]
                pval = background.find_pval(minval,ch1,ch2)
                listtoappend = ((nkevent.interval_datetimes[minvalindex])+half_interval, (np.min(nkevent.correlation_matrix[ch1,ch2])),pval,ch1,ch2)
                min_xcorr_list.append( listtoappend )


    import matplotlib.dates as mdates
    import matplotlib as mpl
    min_date_vals = [items[0] for items in min_xcorr_list]
    min_xcorr_vals =  [items[0] for items in min_xcorr_list]
    #mpl_data = mdates.datetime(min_date_vals)
    fig, ax = plt.subplots(1,1,figsize=(12, 9))
    ax.hist(min_date_vals,color='lightblue',bins=25)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    plt.xticks(rotation=90)
    #plt.show()
    plt.savefig(dirpath+'/datespectra.svg')
    x_corr_effective_plot = np.empty((11,11))
    x_corr_effective_plot[:]=np.nan
    for item in min_xcorr_list:
	pval = item[2]
	timediff = (item[0]-datetime(2009,5,25,0,54,0))
	#print timediff
	#if timedelta(days=-3) < timediff < timedelta(days=3):
	    #Don't data points significantly off date
	    #print "in range"
	x_corr_effective_plot[item[3],item[4]]= pval
	x_corr_effective_plot[item[4],item[3]]= pval

	#else:
	   # x_corr_effective_plot[item[3],item[4]]= np.nan
	   # x_corr_effective_plot[item[4],item[3]]= np.nan
    fig = plt.figure(figsize=(12, 9))
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                               ['green','blue','red'],
                                               261)

    img2 = plt.imshow(x_corr_effective_plot,interpolation='nearest',
                        cmap = cmap2,
                        origin='lower')

    cbar = plt.colorbar(img2,cmap=cmap2)
    
    cbar.set_label('Min P Value for scanned interval')
    fig.savefig(dirpath+'/channeleffacacy.svg')
    return background , nkevent
