#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle
import math

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "# of records in the dataset :", len(enron_data)
print ""
print "# of features for each record:", len(enron_data["SKILLING JEFFREY K"])
print ""
poi_count = 0
for key in enron_data:
    if enron_data[key]['poi']:
        poi_count += 1
print "# of POI :", poi_count
print ""
print "stock belonging to James Prentice :", enron_data["PRENTICE JAMES"]["total_stock_value"]
print ""
print "email messages from Wesley Colwell to persons of interest :", enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print ""
print "value of stock options exercised by Jeffrey K Skilling :", enron_data['SKILLING JEFFREY K']['exercised_stock_options']
print ""
total_payment_skilling = enron_data["SKILLING JEFFREY K"]["total_payments"]
total_payment_kenny = enron_data["LAY KENNETH L"]["total_payments"]
total_payment_fastow = enron_data["FASTOW ANDREW S"]["total_payments"]
print "Payment skilling :", total_payment_skilling
print ""
print "Payment Kenny :", total_payment_kenny
print ""
print "Payment Fastow :", total_payment_fastow
print ""
# How many folks in this dataset have a quantified salary? What about a known email address?
quantified_salary_count = 0
email_address = 0
for key in enron_data:
    if (enron_data[key]['salary']) != 'NaN':
        quantified_salary_count += 1
    if enron_data[key]['email_address'] != 'NaN':
        email_address += 1
print "Quantified Salary count :", quantified_salary_count
print ""
print "Email addresses :", email_address
print ""
total_payments_nan = 0
for key in enron_data:
    if enron_data[key]['total_payments'] == 'NaN':
        total_payments_nan += 1
print "percantage of people with nan in the total payments :", (total_payments_nan*100)/len(enron_data)
