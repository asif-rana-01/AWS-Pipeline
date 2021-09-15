
# Design and create an AWS pipeline for muon detectors

 This project provides a proposal for the architecture of the pipeline for [mdetect](https://mdetect.com.au/). 
 
 The success criteria of this project is based on the creation of an optimal system architecture for the AWS pipeline that is capable of anticipating the usage patterns and provide a cost-effective solution.

 
## Project Requirement 

### Access control for the resources

The users of this project will require to verify through [Multi-Factor Authentication](https://aws.amazon.com/iam/features/mfa/) system before getting access to the resources. 

After successful login to the AWS management console, access to the resources will be determined by the predefined Identity Access Management [IAM](https://aws.amazon.com/iam/) roles.


### Create Database Considering IO Patterns
- ***Storing sensor data generated in real-time***
The IoT devices are capable of generating data
 at a rate of 1KB/second. In order to process 
 this generated data, [Kinesis Data Streams](https://aws.amazon.com/kinesis/data-streams/) will
 be added to the AWS pipeline. The live data 
 stream will then be feed to [Kinesis Data Firehose](https://aws.amazon.com/kinesis/data-firehose/) 
 which will store the data to [S3](https://aws.amazon.com/s3/) buckets for
 storage. 
- ***Storing sensor data in batches*** 
Sensor data might be processed in batches after they have been retrieved from
 the deployed site. This means the pipeline needs to have the capability to 
 store data in batches. [Amazon Simple Storage Service (S3)]((https://aws.amazon.com/s3/)) will be used to store the 
 sensor data. Moreover, this service will be the default storage option for storing the input and output data.  

- ***Database containing information about the sensors***
The deployment status of the IoT sensors is required to be stored in a database that could be accessed and updated accordingly. A SQL based database architecture will be used to store all the required information for the institution. [Amazon Relational Database Service (RDS)](https://aws.amazon.com/rds/) will be able to store this database. 


#### Sample Database

  ![image](https://drive.google.com/uc?export=view&id=1K081OZOpeIAYljiD2SgfijgX6qpNjuzv)

  





