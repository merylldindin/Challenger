# Flask App Deployment on AWS

`Author: Meryll Dindin` | Relative Tutorial on Medium: [link](https://medium.com/@merylldin/from-dev-to-prod-all-you-need-to-know-to-get-your-flask-application-running-on-aws-ecedd4eec55)

## From developement to production:

Hosting a website, getting the appropriate certificates, setting the right listeners to accept HTTPS, here is what it takes with AWS. 

1) **Get the domaine name:**

The first step is simple: Buy an available domain name with **Route 53**. If you wanna run an application as an endpoint and do not really care about having a pretty url to do requests with, you may skip this part.

2) **Setup a running instance with the project:**

Because this project is based on _Flask_, we use the AWS **Elastic Beanstalk** service to run our application online. This service consists in a basic container running on an **EC2** instance and linked to a **S3** bucket for storage. In our case, the application itself does not require a lot of computational power (we do not host our deep learning models there), so we will opt for a **single virtual core** and **4 GB** of RAM (_t2.micro_ instance, which is also the default choice when setting up an instance). Now, AWS generally makes our lives easier as we do not even have to know all of that, or how subnets work, security groups, load balancers, IPs, gateways, and so on. Either you do it online with a clickable solution, or you use the **eb cli**, which is my favorite. AWS provides great [tutorials](https://docs.aws.amazon.com/en_pv/elasticbeanstalk/latest/dg/ebextensions.html), and [templates](https://github.com/awsdocs/elastic-beanstalk-samples/tree/master/configuration-files) to configure your instance. In our case, only one file needs to be defined: `.elasticbeanstalk/config.yml`. (FYI: Pick the `Application` load balancer.) I will redirect anyone to the tutorials for that purpose. One may notice three other files in the project: `.ebextensions/https.config`, which modifies core services to handle the HTTPS requests; `.ebextensions/upgrade_pip.config`, which is a set of command to force _setuptools_ and _pip_ upgrades; `.ebignore`, which is similar to a _.gitignore_ file in the sense that you will only host on the instance that is necessary for it to run properly.

From there, you can already access and visualize your application running online, under a name such as _service.id.zone.aws.com_. However, this is not done through secured connections and this is boring.

3) **Configure the environment variables**

This is mainly a good code practice, but you generally do not want to hardcode your credentials in your code. At least that is widely accepted in production settings. The way to go is to design environment variables, that are easily accessible from the running instance but written nowhere in your code. However, because there is a real difference between a development and a production environment, here are some advices about how to do it properly.

My suggestion would be to use a virtualenv, and to configure your `bin/activate` to incorporate the variables once activated, and unset them when deativated. The way I usually do it is by first creating a json file (that you have to make sure to incorporate in `.ebignore` and `.gitignore`) aggregating all your environment variables.

```python
# Content of environment.json
{
    "SECRET_KEY": "my-secret-key"
}
```

Then use `./setenv.sh dev` and copy-paste the chunks directly into your `bin/activate` file. Once done, deactivate and reactivate your environment. Now you can work easily in your development environment with those environment variables. To verify if those variables have been exported, use:

```bash
python -c "import os, pprint; env_var=os.environ; pprint.pprint(dict(env_var), width=1);"
```

To get back to production settings, the AWS cli makes it as easy as possible. A simple call to `eb setenv key=value` configure your EB instance and the relative environment variable. In the case of this website, we only need to configure **Flask Security** and **Flask Mail**, which is done by calling `./deploy-env.sh prod`. As easy at is sounds.

4) **Get the SSL certificates:**

Unfortunately, your instance cannot use the HTTPS protocol without having SSL certificates. Usually, people would work with **openssl**, which is pretty straightforward. In our case, AWS makes our life easier once again, with the **Certificate Manager** service. You can then create your own certificate based on the domain name you previously bought.

5) **Setup a listener to use HTTPS:**

Once again, everything is as simple as a click. Open the configuration of your **Elastic Beanstalk** instance, and click on _Modify_ under **Load Balancer**. There, add a new **listener** that redirects 443 (HTTPS) to 80 (HTTP) with the appropriate SSL certificate you created just before. For the ones that feel more comfortable with configuration files: see `.ebextensions/listener.config`.

6) **Redirect your domain name to your actual instance:**

Here, we will dive back into the **Route 53**. Two objects have to be configured to redirect the requests: a `Canonical Name`, such as _{cname}.domain-name_ that takes as value your **EB** instance; an `Alias`, such as _{alias}.domain-name_ for your **EB** instance as well. With those two records, you will be good to go!