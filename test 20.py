import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm # Import tqdm
import time
import gc



# GoogLeNet CIFAR-10 호환
class GoogLeNetCIFAR(models.GoogLeNet):
    def __init__(self, num_classes=10):
        # init_weights=False로 설정하여 ImageNet 가중치 로드 시도 안 함
        # CIFAR-10 (32x32)에 맞게 conv1을 변경했으므로, 224x224 이미지용 가중치와 호환되지 않음
        super().__init__(aux_logits=False, num_classes=num_classes, init_weights=False)
        # CIFAR-10 (32x32) 이미지를 직접 처리할 수 있도록 conv1 변경
        # 원본 GoogLeNet은 224x224 입력에 대해 conv1 (7x7, stride 2) 및 maxpool1을 사용
        # CIFAR-10에서는 3x3, stride 1, padding 1이 일반적
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 -> 8x8 (이전 레이어 출력에 따라)

        # Inception 모듈 내부의 maxpool도 고려해야 하지만,
        # 여기서는 모델의 특정 계층만 수정하여 CIFAR-10에 맞춥니다.
        # 이 구현은 ResNet처럼 conv1, maxpool1 후 바로 Inception 블록이 오는 방식입니다.
        # GoogLeNet의 원본 CIFAR-10 포팅은 더 복잡할 수 있습니다.
        # 여기서는 일반적인 작은 이미지 입력에 대한 ResNet-style GoogLeNet 변형으로 간주합니다.

# PlainBlock, PlainNet-34, PReLU-Net
class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = self.relu(out)
        return out

def make_layer(block, in_channels, out_channels, num_blocks, stride):
    layers = [block(in_channels, out_channels, stride)]
    for _ in range(1, num_blocks):
        layers.append(block(out_channels, out_channels))
    return nn.Sequential(*layers)

class PlainNet34(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = make_layer(PlainBlock, 64, 64, 3, 1)
        self.layer2 = make_layer(PlainBlock, 64, 128, 4, 2)
        self.layer3 = make_layer(PlainBlock, 128, 256, 6, 2)
        self.layer4 = make_layer(PlainBlock, 256, 512, 3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ReLU를 PReLU로 재귀적으로 바꾸는 함수
def replace_relu_with_prelu(model):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, nn.PReLU())
        else:
            replace_relu_with_prelu(module)

class PReLUNet34(PlainNet34):
    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        replace_relu_with_prelu(self)

def get_vgg16_bn(num_classes=10):
    model = models.vgg16_bn(weights=None)
    # VGG16_bn의 마지막 분류기 레이어 교체.
    # 기본 VGG의 classifier[6]은 4096개의 입력 특성을 가집니다.
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model

# 학습 및 평가 함수
def train_one_epoch(model, loader, optimizer, criterion, scaler): # scaler 인자 추가
    model.train()
    pbar = tqdm(loader, desc="Training", leave=False) # tqdm 추가
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(): # autocast 활성화
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward() # scaler 사용
        scaler.step(optimizer) # scaler 사용
        scaler.update() # scaler 사용
        pbar.set_postfix(loss=loss.item()) # tqdm에 손실 표시

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    pbar = tqdm(loader, desc="Evaluating", leave=False) # tqdm 추가
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, pred = outputs.topk(5, 1, True, True)
        total += labels.size(0)
        correct = pred.eq(labels.view(-1, 1).expand_as(pred))
        correct_top1 += correct[:, :1].sum().item()
        correct_top5 += correct.sum().item()
    return 100 * correct_top1 / total, 100 * correct_top5 / total

# 메인 실행
def run_all_models_training():
    torch.backends.cudnn.benchmark = True
    global device # 함수 내부에서 device 변수 사용 가능하도록 global 선언
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("사용 디바이스:", device)

    # CIFAR-10 transform (VGG를 위해 224로 리사이즈)
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # 경량 실행을 위한 배치 사이즈 축소 및 num_workers=0 설정
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    model_list = [
        ('VGG-16', get_vgg16_bn(10)),
        ('GoogLeNet', GoogLeNetCIFAR(num_classes=10)),
        ('PReLU-Net', PReLUNet34()),
        ('PlainNet-34', PlainNet34()),
        ('ResNet-34', models.resnet34(num_classes=10)),
        ('ResNet-50', models.resnet50(num_classes=10)),
        ('ResNet-101', models.resnet101(num_classes=10)),
        ('ResNet-152', models.resnet152(num_classes=10)),
    ]

    epochs = 5
    criterion = nn.CrossEntropyLoss()
    results = []
    histories = []

    for name, model in model_list:
        print(f'\n 학습중 {name}')
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        scaler = torch.cuda.amp.GradScaler() # GradScaler 추가

        history = {'epoch': [], 'top1': [], 'top5': [], 'lr': []}
        start_time = time.time()

        for epoch in range(epochs):
            train_one_epoch(model, trainloader, optimizer, criterion, scaler) # scaler 인자 전달
            acc1, acc5 = evaluate(model, testloader)
            print(f'Epoch {epoch+1}/{epochs} - {name} | Top-1: {acc1:.2f}% | Top-5: {acc5:.2f}%')
            history['epoch'].append(epoch + 1)
            history['top1'].append(acc1)
            history['top5'].append(acc5)
            history['lr'].append(scheduler.get_last_lr()[0])
            scheduler.step()

        end_time = time.time()
        print(f'⏱ Training time: {end_time - start_time:.1f} seconds')

        results.append({
            'Model': name,
            'Top-1 Accuracy (%)': acc1,
            'Top-5 Accuracy (%)': acc5,
            'Training Time (sec)': round(end_time - start_time, 2)
        })

        histories.append(history)

        # 정확도 시각화 저장
        plt.figure()
        plt.plot(history['epoch'], history['top1'], label='Top-1')
        plt.plot(history['epoch'], history['top5'], label='Top-5')
        plt.title(f'{name} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid()
        plt.savefig(f'{name}_accuracy_curve.png')
        plt.close()

        # 메모리 정리
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # 최종 결과 출력 및 저장
    print('\n 최종 결과:')
    df = pd.DataFrame(results)
    print(df)
    df.to_csv('cifar10_all_models_results_with_time.csv', index=False)

    # 모든 모델 Top-1 Accuracy 병합 그래프 저장
    plt.figure(figsize=(10, 6))
    for name, history in zip([m[0] for m in model_list], histories):
        plt.plot(history['epoch'], history['top1'], label=name)
    plt.title('Top-1 Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('all_models_top1_accuracy_comparison.png')
    plt.close()

    print("\n✅ 모든 모델 학습 및 병합 그래프, 학습 시간 포함 저장 완료!")

# -------------------------------
if __name__ == '__main__':
    run_all_models_training()