import 'package:flutter/material.dart';

class DetailPage extends StatelessWidget {
  final String title; // 페이지 제목
  final String imagePath; // 이미지 경로
  final String buttonText; // 버튼 텍스트
  final VoidCallback onButtonPressed; // 버튼 클릭 동작

  const DetailPage({
    super.key,
    required this.title,
    required this.imagePath,
    required this.buttonText,
    required this.onButtonPressed,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(title),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Image.asset(imagePath, height: 200, fit: BoxFit.contain),
          const SizedBox(height: 20),
          ElevatedButton(
            onPressed: onButtonPressed,
            child: Text(buttonText),
          ),
        ],
      ),
    );
  }
}