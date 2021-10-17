
# Design and create an AWS pipeline for muon detectors

 This project provides a proposal for the architecture of a pipeline for [mdetect](https://mdetect.com.au/). 
 
 The success criteria of this project is based on the creation of an optimal system architecture for the AWS pipeline that is capable of anticipating the usage patterns and provide a cost-effective solution.

 
## Project Requirement 


<details>
<summary> Access Control for the Resources </summary>
<br>
 
The users of this project will require to verify through [Multi-Factor Authentication](https://aws.amazon.com/iam/features/mfa/) system before getting access to the resources. 

After successful login to the AWS management console, access to the resources will be determined by the predefined- Identity Access Management [IAM](https://aws.amazon.com/iam/) roles.

</details>

<details>
<summary> Create Database Considering IO Patterns </summary>
<br>

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




#### Sample Relational Database

A sample database has been proposed for mDetect which will allow them to manage and track the progress of their ongoing projects.

  ![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/Database%20Schema.png)


</details>


<details>
<summary> Sharing Findings with Clients  </summary>
<br>

The predictions made by the machine learning algorithm will require to be shared with the clients. This requires an API through which the clients will have access to the documents which are needed to be shared. Amazon [API Gateway](https://aws.amazon.com/api-gateway/) can be used to share findings with the clients. Contents in S3 can be shared with the client through a [presigned URL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ShareObjectPreSignedURL.html). 

Moreover, based on certain scenarios, the clients might be required to be notified about certain findings regarding the predictions made by the machine learning model. The functionality of notifying the clients could be implemented by using [Amazon Simple Service Notification (SSN)](https://aws.amazon.com/sns/) service. 
</details>

## Data Preparation
The data generated by the IoT device is stored in JSON format. This requires some data pre-processing to extract the required features from the data and store it into csv format.
Required code to convert the JSON file into CSV can be found [here](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/Conversion-%20JSON%20to%20CSV%20.ipynb).
There were 16 features in the dataset, but the pre-trained machine learning model needed 8 features. Due to this requirement, only the required features were kept in the dataset.

## AWS Implementation: 

### User Access



<details>
<summary> Enabling MFA for Users </summary>

<br>

MFA is enabled for the root account by the following steps-

i)	Selecting My Security Credentials from the top corner drop down menu. 


![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/Screenshot_4.jpg)


![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/mfa-1.jpg)

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/mfa-2.jpg)

 
ii)	Scan the QR code by using a virtual MFA application that supports the TOTP standard.
 
![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/mfa-3.jpg)
 
![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/mfa-4.png)
 
![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/Screenshot_5.jpg)

iii)	After placing the generated codes, the MFA device would be authenticated for the account.

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/Screenshot_6.jpg)

</details>

<details>
<summary> Assigning IAM for Users </summary>
<br>

A user can be created by using the [IAM](https://aws.amazon.com/iam/) feature of AWS.

Steps for creating a user- 

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/iam-1.jpg)

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/iam-2.jpg)

Assigning a user name and credentials

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/iam-3.jpg)

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/iam-4.jpg)

Creating a user group for ease of managing the users

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/iam-5.jpg)

Assigning the roles to the user according to business use

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/iam-6.jpg)

User has been created

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/iam-7.jpg)

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/iam-8.jpg)

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/iam-9.jpg)

</details>


### Access for Clients 

<details>
<summary> Client API Access </summary>
<br>


Whith the help of [presigned URL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ShareObjectPreSignedURL.html), the contents stored in s3 buckets can be shared with the clients for further analysis.

Pre-signed URL for an S3 content can be created by the following steps-

Using the AWS CLI console the directory of the object is passed along with the syntax for pre-signing a document

```bash
  aws s3 presign [s3 directory of the file]
``` 
![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/api-2.jpg)


After the above instruction is given, the CLI console return a link to the file which can then be shared with the client.
![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/api-3.jpg)

Demo: File accessed through presigned URL
![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/api-4.jpg)
</details>

<details>
<summary> Notifying Changes to Clients </summary>
<br>
The clients might be required to be notified based on certain changes predicted by the machine learning algorithm.
The notification can be made through an email which will notify the clients about the recent findings. Amazon SNS is an AWS service that is capable of notifying users through email; if certain changes are being made to the contents of a S3 bucket.
The steps of creating a SNS for a S3 bucket is as follows-


![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/sns-1.jpg)

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/sns-2.jpg)

The policy of the SNS topic is needed to be changed to the name of the S3 bucket.

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/sns-3.jpg)

After the creation of the SNS topic, a subscription is needed to be created to enable communication with the client.

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/sns-4.jpg)

Email address is being configured which will notify the cliensts regarding the project. 

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/sns-5.jpg)

The client needs to confirm the subscription to this SNS topic.

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/sns-6.jpg)

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/sns-7.jpg)

An event notifactaion is being created to nofify the clint of any cahanges in the s3 bucket.

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/sns-s3-8.jpg)

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/sns-s3-9.jpg)

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/sns-s3-10.jpg)

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/sns-s3-11.jpg)

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/sns-s3-12.jpg)

</details>

### Data Storage

<details>
<summary>S3 Bucket</summary>
<br>

As a primary data storage, a S3 bucket is being created-

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/s3-1.jpg)

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/s3-2.jpg)

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/s3-3.jpg)
</details>

<details>
<summary>S3 Glacier Archive</summary>
<br>
S3 Glacier Archive enables storing infrequently accessed data in a cost efficient way.

Steps of configuring a S3 Glacier Archive is as follows-

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/glacier-1.jpg)

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/glacier-2.jpg)

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/glacier-3.jpg)

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/glacier-4.jpg)

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/glacier-5.jpg)
</details>

### Creating Database

<details>
<summary>Amazon RDS</summary>
<br>
Amazon RDS is being selected as the AWS service to create and host the database containing information about the sensors.

The steps are as follows-

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/rds-1.jpg)

![image](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/ref_images/rds-2.jpg)


</details>


### Uploading Model

<details>
<summary>Required Library</summary>
<br>

Python library 'ezsmdeploy' would be used to upload the machine learning model to the AWS cloud platform.  Details about this library can be found [here](https://pypi.org/project/ezsmdeploy/)


</details>


<details>
<summary>Configuring AWS CLI</summary>
<br>

Before uploading the pretrained model, AWS CLI is needed to be configured. AWS Account and Access Keys will be needed to be configured by following this [link](https://docs.aws.amazon.com/powershell/latest/userguide/pstools-appendix-sign-up.html). Later, AWS CLI will be required to be set up by following this [link](https://docs.aws.amazon.com/polly/latest/dg/setup-aws-cli.html)

</details>


<details>
<summary>Hosting model in Sagemaker</summary>
<br>

In order to upload the pretarined model to AWS platform the [model_ezsm.py](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/code_and_model/deploy_ezsm.py) file is needed to be executed on the local device. Before executing the code, the [deploy.py](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/code_and_model/deploy.py) file is needed to be modified accordingly. Through this process, the [model.pickle](https://github.com/asif-rana-01/COS80028-S2-Asif-Rana-102866893/blob/main/code_and_model/model.pickle) file would be uploaded to a S3 bucket.

</details>

<details>
<summary>Creating Model Endpoint</summary>
<br>

An endpoint for the model is needed to be crerated to carry out predictions. Detail procedutre of creating the endpoint for the model can be carried out [here](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateEndpoint.html)

</details>
