# Personal Portfolio Website for me

A modern and responsive personal portfolio website built using HTML, CSS, and JavaScript.

## Features

* Responsive Design
* Animated Typing Effect
* Glowing Profile Image
* Animated Skill Progress Bars
* 3D Hover Effects
* Project Showcase Section
* Resume Download Button
* Social Media Integration
* Professional Footer

## Technologies Used

* HTML5
* CSS3
* JavaScript
* Font Awesome
* AWS S3 Static Website Hosting

## Project Structure

```text
portfolio/
│
├── index.html
├── style.css
├── script.js
├── profile.png
├── resume.pdf
└── README.md
```

## Local Setup

Clone the repository:

```bash
git clone https://github.com/12345ujjwal/github-action-S3-portfolio-project.git
```

Navigate to the project directory:

```bash
cd <repository-name>
```

Open `index.html` in your browser.

---

# Deploying on AWS S3

### Step 1: Create an S3 Bucket

1. Open AWS Management Console.
2. Navigate to S3.
3. Click **Create Bucket**.
4. Enter a unique bucket name.
5. Disable **Block All Public Access**.
6. Create the bucket.

### Step 2: Upload Website Files

Upload all project files:

```text
index.html
style.css
script.js
profile.png
resume.pdf
```

### Step 3: Enable Static Website Hosting

1. Open the bucket.
2. Go to **Properties**.
3. Scroll to **Static Website Hosting**.
4. Click **Edit**.
5. Enable Static Website Hosting.
6. Set:

```text
Index document: index.html
Error document: index.html
```

Save the changes.

### Step 4: Add Bucket Policy

Replace `YOUR_BUCKET_NAME` with your bucket name.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadAccess",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::YOUR_BUCKET_NAME/*"
    }
  ]
}
```

### Step 5: Access Your Website

Navigate to:

```text
Properties → Static Website Hosting
```

Copy the Website Endpoint URL.

Example:

```text
http://your-bucket-name.s3-website-ap-south-1.amazonaws.com
```

Your portfolio is now live.

---

# AWS CLI Deployment (Optional)

Configure AWS CLI:

```bash
aws configure
```

Upload files:

```bash
aws s3 sync . s3://YOUR_BUCKET_NAME
```

Whenever you update your website:

```bash
aws s3 sync . s3://YOUR_BUCKET_NAME --delete
```

---

# Future Improvements

* CloudFront CDN
* HTTPS using ACM Certificate
* Custom Domain with Route 53
* CI/CD using GitHub Actions
* Contact Form Integration
* Visitor Analytics

## Author

**Ujjwal Pratap Singh**

Founder, Skill Nebula Pvt. Ltd.

* Email: [nexaskilllab@gmail.com](mailto:nexaskilllab@gmail.com)
* Website: [www.nexaskilllab.com](http://www.nexaskilllab.com)
* LinkedIn: https://www.linkedin.com/company/skillnebula/

---

⭐ If you found this project useful, consider giving it a star.
