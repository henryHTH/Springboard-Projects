# Importing built-in libraries (no need to install these)
import re
import os
from time import gmtime, strftime, localtime
from datetime import datetime, timedelta
import unicodedata

# Importing libraries you need to install
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import bs4 as bs
from lxml import html
from tqdm import tqdm

def RemoveTags(soup):
    
    '''
    Drops HTML tags, newlines and unicode text from
    filing text.
    
    Parameters
    ----------
    soup : BeautifulSoup object
        Parsed result from BeautifulSoup.
        
    Returns
    -------
    text : str
        Filing text.
        
    '''
    
    # Remove HTML tags with get_text
    text = soup.get_text()
    
    # Remove non acii characters
    text = re.sub('\n|\t|\r|\x0b|\x0c|\x96', ' ', text)
    
    # Replace unicode characters with their
    # "normal" representations
    text = unicodedata.normalize('NFKD', text)
    
    return text


def extract_risk_factors(text):

    '''
    Extrack Item 1A. Risk Factors part by regular expression from 10-K report.
    
    Parameters
    ----------
    text : string
        Parsed and cleaned text from BeautifulSoup.
        
    Returns
    -------
    text : str
        Item 1A. Risk Factors text.
        
    '''

    start,end = -1,-1
    look={" see ", " refer to ", " included in "," contained in ",' in '}
    name_1a = re.compile('Item *1A\. *Risk *Factors|Item *1ARisk *Factors|Item *1A *Risk *Factors', re.IGNORECASE)
    name_1b = re.compile('Item *1B\. *Unresolved *Staff *Comments|Item *1BUnresolved *Staff *Comments|Item *1B.Unresolved *Staff *Comments', re.IGNORECASE)
    name_2 = re.compile('ITEM *2\. *Properties|ITEM *2Properties|ITEM *2 *Properties', re.IGNORECASE)

    first_flag = sum(1 for _ in name_1a.finditer(text,re.IGNORECASE))!=1                
    for m in name_1a.finditer(text,re.IGNORECASE):
        if first_flag == False:
            substring = text[m.start()-20:m.start()]
            if not any(s in substring for s in look):
                if ('Item' in re.split('[\W]+',text[m.end():m.end()+40])) | ('Item' in re.split('[\W]+',text[m.start()-40:m.start()])):
                    continue
                start = m.span()[0]
                #print(start)
                break
        first_flag = False
    
    if start != -1:
        text_end = text[start:]
        for m in name_1b.finditer(text_end,re.IGNORECASE):
            substring = text_end[m.start()-20:m.start()]
            if not any(s in substring for s in look):   
                end = m.span()[0]
                break

        if end != -1:
            return text[start:(start+end)]

        else:
            for m in name_2.finditer(text_end,re.IGNORECASE):
                substring = text_end[m.start()-20:m.start()]
                if not any(s in substring for s in look):   
                	end = m.span()[0]
                	break
            #print(start,end)
            if end != -1:
                return text[start:(start+end)]

            return 
    else:
        return 
    
    
def Extract_Risk_Factors(soup):

    '''
    Extrack Item 1A. Risk Factors part by find the anchor in the web where 10-K report lies.

    Parameters
    ----------
    soup : beautifulsoup object
        Parsed from BeautifulSoup.

    Returns
    -------
    text : str
        Item 1A. Risk Factors text.
    or fundtion extract_risk_factors, if text is none 
    '''

    name_1a,name_1b = '',''
    find_1a = False
    find_1b = False
    href_flag = False
    if len(soup.find_all('a')) != 0: 
        for s in soup.find_all('a'):
            #p_1a = re.compile('Item.1A', re.IGNORECASE)
            #p_1b = re.compile('Item.1B', re.IGNORECASE)
            text = unicodedata.normalize('NFKD', s.text)
            text = re.sub('\n|\t|\r|\x0b|\x0c|\x96', ' ', text)
            if not find_1a:
                if re.search('Item *1A|Risk *Factors|Item *1A\. *Risk *Factors',text.strip(), re.IGNORECASE):
                #(re.match('Item 1A',text.strip())) or (re.match('Risk Factors',text.strip(), re.IGNORECASE)) or (re.match('Item *1A\. *Risk *Factors',text.strip(), re.IGNORECASE)) :
                    if 'href' in s.attrs:
                        name_1a = s['href'][1:]
                        find_1a = True
            if not find_1b:
                if re.search('Item *1B|Unresolved *Staff *Comments|Item *1B\. *Unresolved *Staff *Comments',text.strip(), re.IGNORECASE):
                #(re.match('Item 1B',text.strip())) or (re.match('Unresolved Staff Comments',text.strip(), re.IGNORECASE)) or (re.match('Item *1B\. *Unresolved *Staff *Comments',text.strip(), re.IGNORECASE)):
                    if 'href' in s.attrs:
                        name_1b = s['href'][1:]
                        find_1b = True
            if find_1a & find_1b: 
                href_flag = True
                break

        if href_flag:
            for name in ['name','id']:
            	for tag in ['a','div']:
	                a = str(soup).find(f'<{tag} {name}="{name_1a}">',re.IGNORECASE)
	                b = str(soup).find(f'<{tag} {name}="{name_1b}">',re.IGNORECASE)
	                if (a==-1):
	                    a = str(soup).lower().find(f'<{tag} {name}="{name_1a}">'.lower(),re.IGNORECASE)
	                if (b==-1):
	                    b = str(soup).lower().find(f'<{tag} {name}="{name_1b}">'.lower(),re.IGNORECASE)
	                if (a!=-1) & (b!=-1):
	                    risk_file = str(soup)[a:b]
	                    cleansoup = bs.BeautifulSoup(risk_file, "lxml")
	                    # Drops HTML tags, newlines and unicode text from filing text.
	                    risk_text = RemoveTags(cleansoup)
	                    #print(risk_text)
	                    return risk_text
	                else:
	                    continue
            #print('Cant find a match')
            return extract_risk_factors(RemoveTags(bs.BeautifulSoup(soup.text, 'lxml')))
        else:
            #print('Cant find a match')
            return  extract_risk_factors(RemoveTags(bs.BeautifulSoup(soup.text, 'lxml')))
    else:
        #print('Cant find a match')
        return extract_risk_factors(RemoveTags(bs.BeautifulSoup(soup.text, 'lxml')))


def Scrape10K(browse_url_base, filing_url_base, doc_url_base, cik, acc_no, log_file_name):
    
    '''
    Scrapes all 10-Ks and 10-K405s for a particular 
    CIK from EDGAR.
    
    Parameters
    ----------
    browse_url_base : str
        Base URL for browsing EDGAR.
    filing_url_base : str
        Base URL for filings listings on EDGAR.
    doc_url_base : str
        Base URL for one filing's document tables
        page on EDGAR.
    cik : str
        Central Index Key.
    log_file_name : str
        Name of the log file (should be a .txt file).
        
    Returns
    -------
    None.
    
    '''

    # Check if we've already scraped this CIK
    try:
        os.mkdir(cik)
    except OSError:
       	print("Already scraped CIK", cik)
    #    return
    
    # If we haven't, go into the directory for that CIK
    os.chdir(cik)
    print('Scraping CIK', cik)
    

    # Get the accession number for the filing
    #acc_no = str(row['Acc_No'])
    #date = str(row['Filing Date'])
    # Navigate to the page for the filing
    docs_page = requests.get(filing_url_base % (cik, acc_no))

    # If request fails, log the failure
    # and skip to the next filing
    if docs_page.status_code != 200:
        os.chdir('..')
        #text = "Request failed with error code " + str(docs_page.status_code) + \
        #       "\nFailed URL: " + (filing_url_base % (cik, acc_no, date)) + '\n'
        #WriteLogFile(log_file_name, text)
        #os.chdir(cik)
        return

    # If request succeeds, keep going...

    # Parse the table of documents for the filing
    docs_page_soup = bs.BeautifulSoup(docs_page.text, 'lxml')
    docs_html_tables = docs_page_soup.find_all('table')
    if len(docs_html_tables)==0:
        return 
    docs_table = pd.read_html(str(docs_html_tables[0]), header=0)[0]
    docs_table['Type'] = [str(x) for x in docs_table['Type']]

    # Get the 10-K and 10-K405 entries for the filing
    docs_table = docs_table[(docs_table['Type'] == '10-K')]

    # If there aren't any 10-K or 10-K405 entries,
    # skip to the next filing
    if len(docs_table)==0:
        return 
    # If there are 10-K or 10-K405 entries,
    # grab the first document
    elif len(docs_table)>0:
        docs_table = docs_table.iloc[0]

    docname = docs_table['Document']
    if 'iXBRL' in docname:
        docname = docname.split(' ')[0]


    date = [x.text for x in docs_page_soup.find_all('div',{"class": "info"})][-1]
    if type(date) != str:
        date = 'not_available'

    # If that first entry is unavailable,
    # log the failure and exit
    if str(docname) == 'nan':
        os.chdir('..')
        #text = 'File with CIK: %s and Acc_No (date: %s): %s is unavailable' % (cik, date, acc_no) + '\n'
        #WriteLogFile(log_file_name, text)
        #os.chdir(cik)
        return        

    if '.txt' in docname:
	    filename = cik + '_' + date + '_patched.txt'
	    if os.path.exists(filename):
	    	print('file already exits')
	    	os.chdir('..')
	    	return 
    else:
	    filename = cik + '_' + date + '_html_patched.txt'
	    if os.path.exists(filename):
	        print('file already exits')
	        os.chdir('..')
	        return 
    # If it is available, continue...

    # Request the file
    file = requests.get(doc_url_base % (cik, acc_no.replace('-', ''), docname))

    # If the request fails, log the failure and exit
    if file.status_code != 200:
        os.chdir('..')
        #text = "Request failed with error code " + str(file.status_code) + \
        #       "\nFailed URL: " + (doc_url_base % (cik, acc_no.replace('-', ''), docname)) + '\n'
        #WriteLogFile(log_file_name, text)
        #os.chdir(cik)
        return

    # If it succeeds, keep going...
    # Sparse 10-K report 
    file_soup = bs.BeautifulSoup(file.text, 'lxml')

    # If flag is true, means extract_risk_factors function failed to get correct data, need further analysis
    #if need_file is True:
    #    print(file.url)
    # Extract Item 1A Risk Factors 
    risk_file = Extract_Risk_Factors(file_soup)

    if risk_file is None:
        os.chdir('..')
        text = 'Risk Factors section with CIK: %s and Acc_No (date: %s): %s is unavailable' % (cik, date, acc_no) + '\n'
        WriteLogFile(log_file_name, text)
        #os.chdir(cik)
        return 


    # Save the file in appropriate format
    
    html_file = open(filename, 'a')
    html_file.write(risk_file)
    html_file.close()

    # Move back to the main 10-K directory
    os.chdir('..')
    return


def WriteLogFile(log_file_name, text):
    
    '''
    Helper function.
    Writes a log file with all notes and
    error messages from a scraping "session".
    
    Parameters
    ----------
    log_file_name : str
        Name of the log file (should be a .txt file).
    text : str
        Text to write to the log file.
        
    Returns
    -------
    None.
    
    '''
    
    with open(log_file_name, "a") as log_file:
        log_file.write(text)

    return



if __name__ == '__main__':
	
	table = pd.read_csv(r'missing_data.csv').astype('str')
	ciks = table.iloc[:,5].apply(lambda x: x.split('.')[0].zfill(10))
	acc_nos = table.iloc[:-1,10]

	pathname = r'/Users/henry/Documents/study/Springboard/nlp_project'
	data_pathname = r'/Users/henry/Documents/study/Springboard/nlp_project/data'

	os.chdir(pathname)


	# Run the function to scrape 10-Ks

	# Define parameters
	browse_url_base_10k = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=%s&type=10-K'
	filing_url_base_10k = 'http://www.sec.gov/Archives/edgar/data/%s/%s-index.html'
	doc_url_base_10k = 'http://www.sec.gov/Archives/edgar/data/%s/%s/%s'

	# Set correct directory
	os.chdir(data_pathname)

	# Initialize log file
	# (log file name = the time we initiate scraping session)
	#time = strftime("%Y-%m-%d %Hh%Mm%Ss", gmtime())
	


	
	time = strftime("%y%y-%m-%d,%H:%M:%S", localtime())
	log_file_name = 'log '+time+'.txt'
	with open(log_file_name, 'a') as log_file:
	    log_file.write(time + '\n')
	    log_file.close()

	'''
	Scrape10K(browse_url_base=browse_url_base_10k, 
	          filing_url_base=filing_url_base_10k, 
	          doc_url_base=doc_url_base_10k,
	          cik='0000711669', 
	          acc_no='0001140361-13-011986',
	          log_file_name=log_file_name)	
	'''
	# Iterate over CIKs and scrape 10-Ks
	
	for cik,acc_no in tqdm(zip(ciks[467:],acc_nos[467:]),total=len(ciks[467:])):
	    Scrape10K(browse_url_base=browse_url_base_10k, 
	          filing_url_base=filing_url_base_10k, 
	          doc_url_base=doc_url_base_10k,
	          cik=cik, 
	          acc_no=acc_no,
	          log_file_name=log_file_name)

	with open(log_file_name, 'a') as log_file:
	    log_file.write(strftime("%y%y-%m-%d,%H:%M:%S", localtime()) + '\n')
	    log_file.close()
	
