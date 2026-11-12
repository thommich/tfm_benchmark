#!/usr/bin/env bash
set -euo pipefail

# Run in Linux CI/CodeBuild with Docker, AWS CLI and an authenticated sa-east-1 role.
: "${AWS_ACCOUNT_ID:?Set the AWS account ID}"
: "${ECR_REPOSITORY:?Set the ECR repository name}"
region=sa-east-1
image="${AWS_ACCOUNT_ID}.dkr.ecr.${region}.amazonaws.com/${ECR_REPOSITORY}"
tag="baseline-$(git rev-parse --short=12 HEAD)"

aws ecr get-login-password --region "$region" |
  docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${region}.amazonaws.com"
docker build --platform linux/amd64 -f cloud/Dockerfile -t "$image:$tag" .
docker run --rm --entrypoint python "$image:$tag" -m tabtune_baseline.image_self_test
docker push "$image:$tag"
digest=$(aws ecr describe-images --region "$region" --repository-name "$ECR_REPOSITORY" \
  --image-ids "imageTag=$tag" --query 'imageDetails[0].imageDigest' --output text)
test "$digest" != None
printf '%s@%s\n' "$image" "$digest"
