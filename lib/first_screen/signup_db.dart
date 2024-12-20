import 'package:flutter/material.dart';
import 'tops_details.dart';


class SignupDBPage extends StatelessWidget {
  final String userId;
  const SignupDBPage({super.key, required this.userId});

  // 옷 블록 데이터
  static const List<Map<String, String>> clothingItems = [
    {'label': '긴팔상의', 'imagePath': 'assets/images/long_sleeve.png'},
    {'label': '반팔상의', 'imagePath': 'assets/images/short_sleeve.png'},
    {'label': '셔츠', 'imagePath': 'assets/images/shirt.png'},
    {'label': '민소매', 'imagePath': 'assets/images/sleeveless.png'},
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text('상의'),
        backgroundColor: Colors.white,
        foregroundColor: Colors.black,
      ),
      body: GridView.builder(
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 3, // 한 줄에 3개의 블록
          crossAxisSpacing: 10,
          mainAxisSpacing: 10,
        ),
        padding: const EdgeInsets.all(16),
        itemCount: clothingItems.length,
        itemBuilder: (context, index) {
          final item = clothingItems[index];
          return GestureDetector(
            onTap: () {
              // 상세 페이지로 이동하며 데이터 전달
              Navigator.push(
                context,
                PageRouteBuilder(
                  pageBuilder: (context, animation1, animation2) => TopsDetailPage(
                    label: item['label']!,
                    imagePath: item['imagePath']!,
                    userId: userId,
                  ),
                  transitionDuration: Duration.zero, // 모션 제거
                  reverseTransitionDuration: Duration.zero, // 모션 제거
                ),
              );
            },
            child: buildClothingBlock(item['label']!, item['imagePath']!),
          );
        },
      ),
    );
  }

  Widget buildClothingBlock(String label, String imagePath) {
    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: Colors.grey, width: 1.0),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(8.0),
              child: Image.asset(
                imagePath,
                fit: BoxFit.contain,
              ),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            label,
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
}
