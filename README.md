## End to End MAchine Learning Project

1. Docker Build checked
2. Github Workflow
3. Iam User In AWS

## Docker Setup In EC2 commands to be Executed

#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

## Configure EC2 as self-hosted runner:

## Setup github secrets:

AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

ECR_REPOSITORY_NAME = simple-app


## Deploy end to end ML application to AWS EKS
## Step-1
Build a Docker Image for the Streamlit Application, push it to dockerhub

## Step-2
Creating the Amazon Elastic Kubernetes Service(EKS) Service(Using CLI)
	a. Install AWS CLI and then add Secret Access Keys and Access Keys
	b. Download Kubernetes CLI(https://s3.us-west-2.amazonaws.com/amazon-eks/1.23.7/2022-06-29/bin/windows/amd64/kubectl.exe), its basically a CLI tool to work with EKS directly.
	c. Install eksctl (choco install -y eksctl), its CLI tool that will enable you to create an Amazon EKS cluster easier and faster. A single `eksctl` command will create an Amazon EKS cluster with all the resources.
	d. Creating the Amazon EKS Cluster using eksctl, eg. eksctl create cluster --name studentperformance --region us-east-2 --with-oidc --ssh-access --ssh-public-key sample-ec2 --instance-types=m5.xlarge --managed 
	e. Deploying the streamlit app, for your ref. https://github.com/mickey2798/end-to-end-ML-Project/blob/main/streamlit-app-deployment.yaml
	f. deployment cmnd - kubectl apply -f streamlit-app-deployment.yaml
	g. Viewing the deployed resource - kubectl get pods
	h. View all the deployments - kubectl get deployments
	i. View the service - kubectl get services, this cmd exposes the containerized application on an `EXTERNAL IP` address. You can access the application using the given URL. 
	


My deployed application Link - http://abd598939ace542babd57e65c8e9bd63-774368792.us-east-2.elb.amazonaws.com/
