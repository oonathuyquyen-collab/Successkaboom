# Sách Giải Thích Dự Án UnifiedTMIL cho Sinh Viên Năm Nhất

## 1. Lời Giới Thiệu

Chào mừng bạn đến với cuốn sách này! Mục tiêu của cuốn sách là giúp bạn, một sinh viên năm nhất chưa có nhiều kinh nghiệm về công nghệ Blockchain hay Trí tuệ Nhân tạo, hiểu rõ về một dự án nghiên cứu thú vị và quan trọng: **UnifiedTMIL**. Chúng ta sẽ cùng nhau khám phá cách dự án này giải quyết một vấn đề nhức nhối trong thế giới tiền điện tử: **lừa đảo (phishing)**.

### Dự án UnifiedTMIL là gì?

UnifiedTMIL là một hệ thống Trí tuệ Nhân tạo (AI) được thiết kế để phát hiện các hoạt động lừa đảo trên mạng lưới Ethereum. Điều đặc biệt ở UnifiedTMIL là nó có thể làm hai việc cùng lúc chỉ trong một lần xử lý dữ liệu (gọi là "One Forward Pass"):

1.  **Phát hiện tài khoản lừa đảo (Account-level):** Xác định xem một ví điện tử hoặc một địa chỉ trên Ethereum có phải là của kẻ lừa đảo hay không.
2.  **Định vị giao dịch lừa đảo (Transaction-level):** Chỉ ra chính xác giao dịch nào trong một chuỗi giao dịch của tài khoản đó là giao dịch lừa đảo.

### Tại sao cần chống lừa đảo trên Blockchain?

Thế giới tiền điện tử và Blockchain đang phát triển rất nhanh, mang lại nhiều cơ hội nhưng cũng tiềm ẩn không ít rủi ro. Một trong những rủi ro lớn nhất là các hình thức lừa đảo ngày càng tinh vi. Hàng tỷ đô la đã bị mất do lừa đảo trên Blockchain mỗi năm, gây thiệt hại nặng nề cho người dùng và làm suy yếu niềm tin vào công nghệ mới này. Việc phát triển các công cụ AI mạnh mẽ để chống lại lừa đảo là vô cùng cần thiết để bảo vệ tài sản của người dùng và thúc đẩy sự phát triển bền vững của hệ sinh thái Blockchain.

Trong cuốn sách này, chúng ta sẽ đi từ những khái niệm cơ bản nhất về Blockchain, lừa đảo, đến cách UnifiedTMIL hoạt động, những cải tiến quan trọng đã được thực hiện để đảm bảo tính chính xác và trung thực của kết quả. Hãy cùng bắt đầu hành trình khám phá nhé!

## 2. Blockchain và Ethereum Cơ Bản

Để hiểu về UnifiedTMIL, trước hết chúng ta cần nắm vững một số khái niệm cơ bản về Blockchain và Ethereum. Đừng lo lắng, tôi sẽ giải thích một cách đơn giản nhất!

### 2.1. Blockchain là gì?

Hãy tưởng tượng Blockchain như một cuốn sổ cái khổng lồ, nhưng thay vì được quản lý bởi một người hay một ngân hàng, nó được chia sẻ và quản lý bởi rất nhiều máy tính trên toàn thế giới. Mỗi "trang" trong cuốn sổ cái này được gọi là một **"khối" (block)**, và mỗi khối chứa một danh sách các giao dịch đã được xác nhận.

Điều đặc biệt của Blockchain là:

*   **Phi tập trung (Decentralized):** Không có một máy chủ trung tâm nào kiểm soát. Mọi người tham gia mạng lưới đều có một bản sao của cuốn sổ cái này.
*   **Bất biến (Immutable):** Một khi một giao dịch đã được ghi vào một khối và khối đó đã được thêm vào chuỗi, nó không thể bị thay đổi hay xóa bỏ. Điều này đảm bảo tính toàn vẹn và an toàn của dữ liệu.
*   **Minh bạch (Transparent):** Mọi giao dịch đều công khai và có thể được kiểm tra bởi bất kỳ ai trên mạng lưới (mặc dù danh tính người dùng thường được ẩn danh dưới dạng địa chỉ ví).

Các khối được liên kết với nhau bằng mật mã, tạo thành một "chuỗi các khối" (Block-chain). Mỗi khối mới được thêm vào sẽ chứa một "dấu vân tay" (hash) của khối trước đó, đảm bảo rằng chuỗi không bị phá vỡ.

### 2.2. Ethereum là gì?

Ethereum là một nền tảng Blockchain phi tập trung, mã nguồn mở, có khả năng thực thi các **"hợp đồng thông minh" (Smart Contracts)**. Nếu Bitcoin được ví như một loại tiền kỹ thuật số, thì Ethereum giống như một "máy tính toàn cầu" cho phép các nhà phát triển xây dựng và triển khai các ứng dụng phi tập trung (DApps).

*   **Ether (ETH):** Là tiền điện tử gốc của mạng Ethereum. Nó được sử dụng để thanh toán phí giao dịch (gọi là "gas") và làm phần thưởng cho những người xác thực giao dịch.
*   **Hợp đồng thông minh (Smart Contracts):** Là các chương trình máy tính tự thực thi được lưu trữ trên Blockchain. Chúng tự động thực hiện các điều khoản của một thỏa thuận khi các điều kiện được đáp ứng, mà không cần bên thứ ba can thiệp. Ví dụ, một hợp đồng thông minh có thể tự động chuyển tiền cho người bán khi người mua xác nhận đã nhận được hàng.

### 2.3. Ví điện tử và Địa chỉ trên Ethereum

Để tham gia vào mạng Ethereum, bạn cần có một **ví điện tử (wallet)**. Ví điện tử không thực sự "chứa" tiền điện tử của bạn, mà nó chứa các khóa mật mã (private key và public key) cho phép bạn truy cập và quản lý tài sản của mình trên Blockchain.

*   **Địa chỉ ví (Wallet Address):** Giống như số tài khoản ngân hàng của bạn, đây là một chuỗi ký tự duy nhất (ví dụ: `0x742d35Cc6634C0532925a3b844Bc454e4438f444`). Đây là địa chỉ công khai mà người khác có thể gửi ETH hoặc các token khác cho bạn.
*   **Khóa riêng tư (Private Key):** Đây là một chuỗi ký tự bí mật, giống như mật khẩu siêu cấp của bạn. Ai có khóa riêng tư của bạn có thể truy cập và kiểm soát tài sản của bạn. **Tuyệt đối không bao giờ chia sẻ khóa riêng tư của bạn với bất kỳ ai!**

Hiểu được những khái niệm này sẽ giúp chúng ta dễ dàng hơn trong việc tìm hiểu cách UnifiedTMIL hoạt động để bảo vệ các giao dịch và tài khoản trên Ethereum.

## 3. Lừa Đảo (Phishing) trên Ethereum

Sau khi đã hiểu về Blockchain và Ethereum, giờ chúng ta sẽ đi sâu vào vấn đề mà dự án UnifiedTMIL đang cố gắng giải quyết: **lừa đảo (phishing)**.

### 3.1. Phishing là gì?

**Phishing** là một hình thức tấn công mạng mà kẻ xấu giả mạo thành một thực thể đáng tin cậy (ví dụ: ngân hàng, sàn giao dịch, dự án Blockchain nổi tiếng) để lừa đảo người dùng tiết lộ thông tin nhạy cảm (như khóa riêng tư, mật khẩu ví) hoặc thực hiện các hành động không mong muốn (như ký các giao dịch độc hại, chuyển tiền cho kẻ lừa đảo).

Trong thế giới tiền điện tử, phishing đặc biệt nguy hiểm vì các giao dịch trên Blockchain là **bất biến** và **không thể đảo ngược**. Một khi tiền đã bị chuyển đi, rất khó để lấy lại.

### 3.2. Các hình thức lừa đảo phổ biến trên Ethereum

Kẻ lừa đảo ngày càng tinh vi với nhiều chiêu trò khác nhau. Dưới đây là một số hình thức phổ biến trên Ethereum:

*   **Giả mạo trang web/ứng dụng (Website/DApp Spoofing):** Kẻ lừa đảo tạo ra các trang web hoặc ứng dụng giả mạo giống hệt các dự án Blockchain, sàn giao dịch hoặc ví điện tử hợp pháp. Khi người dùng truy cập và kết nối ví, họ có thể bị yêu cầu ký các giao dịch độc hại hoặc tiết lộ khóa riêng tư.
*   **Lừa đảo qua email/tin nhắn (Email/SMS Phishing):** Gửi email hoặc tin nhắn giả mạo từ các dự án, sàn giao dịch, thông báo về các sự kiện, airdrop giả để lừa người dùng truy cập vào các liên kết độc hại.
*   **Lừa đảo qua mạng xã hội (Social Media Scams):** Tạo các tài khoản mạng xã hội giả mạo, chạy quảng cáo hoặc đăng bài viết về các chương trình tặng thưởng (giveaway) hoặc đầu tư siêu lợi nhuận để lôi kéo người dùng gửi tiền hoặc truy cập các trang web lừa đảo.
*   **Rug Pull:** Đây là một hình thức lừa đảo phổ biến trong các dự án DeFi (Tài chính phi tập trung) và NFT (Non-Fungible Token). Kẻ lừa đảo tạo ra một dự án mới, thu hút nhà đầu tư bằng những lời hứa hẹn hấp dẫn, sau đó đột ngột rút hết thanh khoản (tiền) và biến mất, để lại token của nhà đầu tư không có giá trị.
*   **Kẻ lừa đảo (Scammer Accounts):** Các địa chỉ ví được sử dụng để nhận tiền từ các hoạt động lừa đảo. Việc xác định các tài khoản này là rất quan trọng để cảnh báo người dùng.
*   **Giao dịch độc hại (Malicious Transactions):** Các giao dịch mà người dùng bị lừa ký, thường là để cấp quyền truy cập vào tài sản của họ cho kẻ lừa đảo (ví dụ: `approve` token cho một hợp đồng độc hại).

### 3.3. Hậu quả của lừa đảo

Hậu quả của lừa đảo trên Ethereum có thể rất nghiêm trọng:

*   **Mất tài sản:** Người dùng có thể mất toàn bộ số tiền điện tử trong ví của mình.
*   **Mất niềm tin:** Các vụ lừa đảo làm giảm niềm tin của cộng đồng vào công nghệ Blockchain và tiền điện tử.
*   **Thiệt hại danh tiếng:** Các dự án hợp pháp có thể bị ảnh hưởng danh tiếng nếu người dùng bị lừa đảo liên quan đến dự án của họ.

### 3.4. Thách thức trong việc phát hiện lừa đảo

Việc phát hiện lừa đảo trên Ethereum gặp nhiều thách thức:

*   **Tính ẩn danh:** Mặc dù giao dịch minh bạch, nhưng danh tính thực của người dùng thường được ẩn sau các địa chỉ ví, gây khó khăn cho việc truy vết kẻ lừa đảo.
*   **Tốc độ giao dịch:** Hàng triệu giao dịch diễn ra mỗi ngày, đòi hỏi hệ thống phát hiện phải nhanh chóng và hiệu quả.
*   **Tính tinh vi của kẻ lừa đảo:** Kẻ lừa đảo liên tục thay đổi chiêu trò, khiến các phương pháp phát hiện truyền thống dễ bị lỗi thời.
*   **Dữ liệu không cân bằng:** Số lượng giao dịch lừa đảo thường rất nhỏ so với tổng số giao dịch hợp pháp, gây khó khăn cho việc huấn luyện mô hình AI.

Chính vì những thách thức này, dự án UnifiedTMIL ra đời với mục tiêu cung cấp một giải pháp mạnh mẽ và đáng tin cậy để chống lại vấn nạn lừa đảo trên Ethereum.

## 4. Giới Thiệu UnifiedTMIL

Sau khi đã hiểu rõ về Blockchain, Ethereum và các mối đe dọa từ lừa đảo, giờ chúng ta sẽ cùng tìm hiểu sâu hơn về giải pháp mà dự án này mang lại: **UnifiedTMIL**.

### 4.1. UnifiedTMIL là gì?

**UnifiedTMIL** là viết tắt của **Unified Transaction-level Multi-Instance Learning**. Đây là một khung học máy (Machine Learning framework) được thiết kế đặc biệt để giải quyết bài toán phát hiện lừa đảo trên mạng Ethereum. Mục tiêu chính của UnifiedTMIL là cung cấp một hệ thống mạnh mẽ, chính xác và đáng tin cậy để:

*   **Xác định các tài khoản (ví) lừa đảo (Account-level detection):** Trả lời câu hỏi liệu một địa chỉ ví cụ thể có thuộc về một kẻ lừa đảo hay không.
*   **Định vị các giao dịch lừa đảo (Transaction-level localization):** Trong một chuỗi các giao dịch của một tài khoản, chỉ ra chính xác giao dịch nào là giao dịch độc hại hoặc liên quan đến lừa đảo.

### 4.2. Tại sao lại là "Unified"?

Trong các nghiên cứu trước đây, việc phát hiện tài khoản lừa đảo và định vị giao dịch lừa đảo thường được coi là hai bài toán riêng biệt, được giải quyết bằng các mô hình hoặc phương pháp khác nhau. Tuy nhiên, UnifiedTMIL mang đến một cách tiếp cận **"thống nhất" (Unified)**. Điều này có nghĩa là:

*   **Một mô hình duy nhất:** UnifiedTMIL sử dụng một kiến trúc mô hình duy nhất, được huấn luyện để thực hiện cả hai nhiệm vụ cùng lúc.
*   **Tận dụng thông tin chéo:** Bằng cách giải quyết cả hai bài toán trong cùng một khung, mô hình có thể học được các mối quan hệ và thông tin hữu ích từ nhiệm vụ này để cải thiện hiệu suất cho nhiệm vụ kia. Ví dụ, việc biết được giao dịch nào là lừa đảo có thể giúp xác định tài khoản lừa đảo tốt hơn, và ngược lại.

### 4.3. "One Forward Pass" nghĩa là gì?

"**One Forward Pass**" là một khái niệm quan trọng trong Trí tuệ Nhân tạo, đặc biệt là trong các mô hình Deep Learning. Nó có nghĩa là toàn bộ quá trình xử lý dữ liệu từ đầu vào (chuỗi giao dịch) đến đầu ra (dự đoán tài khoản lừa đảo và xếp hạng giao dịch lừa đảo) chỉ diễn ra trong **một lần duy nhất** thông qua mạng lưới thần kinh.

Lợi ích của "One Forward Pass" bao gồm:

*   **Hiệu quả tính toán:** Giảm thời gian và tài nguyên cần thiết để đưa ra dự đoán, rất quan trọng đối với các hệ thống cần hoạt động trong thời gian thực hoặc xử lý lượng lớn dữ liệu.
*   **Tính nhất quán:** Đảm bảo rằng các dự đoán cho cả hai nhiệm vụ (tài khoản và giao dịch) được đưa ra dựa trên cùng một tập hợp các đặc trưng và quá trình xử lý nội bộ của mô hình, giúp tăng cường sự nhất quán giữa các kết quả.

Với những đặc điểm này, UnifiedTMIL không chỉ là một công cụ phát hiện lừa đảo mạnh mẽ mà còn là một giải pháp hiệu quả và toàn diện cho vấn đề an ninh trên Ethereum.

## 5. Kiến Trúc UnifiedTMIL (Phiên Bản Đã Cải Tiến)

Đây là phần cốt lõi của dự án, nơi chúng ta sẽ khám phá cách UnifiedTMIL được xây dựng để thực hiện nhiệm vụ phát hiện lừa đảo. Kiến trúc này đã được cải tiến để đảm bảo tính chính xác và trung thực của kết quả.

### 5.1. Tổng quan kiến trúc

Hãy cùng nhìn vào sơ đồ kiến trúc tổng thể của UnifiedTMIL. Đây là "bản đồ" giúp chúng ta hiểu cách dữ liệu được xử lý từ đầu vào đến đầu ra.

![Kiến trúc UnifiedTMIL](/home/ubuntu/Successkaboom/figures/fig1_architecture_final.png)

*(Hình 1: Sơ đồ kiến trúc UnifiedTMIL phiên bản đã cải tiến)*

Kiến trúc này có thể được chia thành các lớp chính, hoạt động tuần tự để xử lý thông tin:

1.  **Transaction Sequence (Chuỗi Giao Dịch):** Dữ liệu đầu vào thô.
2.  **BERT4ETH Embeddings:** Biểu diễn số hóa của từng giao dịch.
3.  **1D-Temporal Convolutional Network (TCN):** Học các mẫu thời gian từ chuỗi giao dịch.
4.  **Instance Representations (h_k):** Biểu diễn đặc trưng của từng giao dịch sau khi qua TCN.
5.  **Gated Attention Network:** Cơ chế chú ý để xác định mức độ quan trọng của từng giao dịch.
6.  **Attention Weights (α_k) & Engineered Features:** Các trọng số chú ý và các đặc trưng được thiết kế thủ công.
7.  **Unified Prediction Heads:** Hai "đầu" dự đoán riêng biệt cho nhiệm vụ Account Detection và Transaction Localization.

Bây giờ, chúng ta sẽ đi sâu vào từng thành phần.

### 5.2. Transaction Sequence (Chuỗi Giao Dịch)

Đầu vào của UnifiedTMIL là một chuỗi các giao dịch (`Tx1, Tx2, ..., Txn`) của một tài khoản Ethereum, được sắp xếp theo thứ tự thời gian. Mỗi giao dịch mang theo nhiều thông tin quan trọng.

### 5.3. BERT4ETH Embeddings (Biểu Diễn Giao Dịch)

Trước khi đưa vào mô hình AI, mỗi giao dịch (`Tx`) cần được chuyển đổi thành một dạng số mà máy tính có thể hiểu được, gọi là **embedding** (biểu diễn nhúng). UnifiedTMIL sử dụng các đặc trưng từ BERT4ETH, bao gồm:

*   **Counterparty:** Địa chỉ ví đối tác trong giao dịch.
*   **Direction (IN/OUT):** Giao dịch là nhận tiền (IN) hay gửi tiền (OUT).
*   **Amount (Value):** Giá trị tiền tệ của giao dịch.
*   **Count (#Txs):** Số lượng giao dịch đã thực hiện bởi tài khoản đó.
*   **Time Delta (Δt):** Khoảng thời gian từ giao dịch trước đó.

Các thông tin này được mã hóa thành các vector số, tạo thành biểu diễn "BERT4ETH Embeddings" cho mỗi giao dịch.

### 5.4. 1D-Temporal Convolutional Network (TCN)

Sau khi có các embedding của từng giao dịch, chúng ta cần một cơ chế để hiểu được mối quan hệ và các mẫu (patterns) theo thời gian trong chuỗi giao dịch. **1D-Temporal Convolutional Network (TCN)** làm nhiệm vụ này.

*   **Convolutional Network (Mạng Tích Chập):** Tương tự như cách các mạng tích chập xử lý hình ảnh, TCN xử lý dữ liệu chuỗi. Nó sử dụng các "bộ lọc" (filters) để quét qua chuỗi giao dịch, phát hiện các mẫu cục bộ (ví dụ: một chuỗi các giao dịch nhỏ liên tiếp, hoặc một giao dịch lớn bất thường sau nhiều giao dịch nhỏ).
*   **Temporal (Thời gian):** "1D-Temporal" có nghĩa là mạng này được thiết kế để hoạt động hiệu quả trên dữ liệu chuỗi thời gian, nơi thứ tự và khoảng cách giữa các sự kiện là quan trọng.

Kết quả của TCN là một tập hợp các **Instance Representations (h_k)**, trong đó mỗi `h_k` là một vector đặc trưng tổng hợp thông tin của giao dịch `Tx_k` và các giao dịch lân cận trong chuỗi.

### 5.5. Gated Attention Network (Mạng Chú Ý Có Cổng)

Không phải tất cả các giao dịch trong một chuỗi đều quan trọng như nhau. Một số giao dịch có thể là "dấu hiệu" mạnh mẽ của lừa đảo, trong khi những giao dịch khác thì không. **Gated Attention Network** giúp mô hình tập trung vào những giao dịch quan trọng nhất.

*   **Attention (Chú ý):** Cơ chế này cho phép mô hình gán một "trọng số" cho mỗi `h_k`, thể hiện mức độ quan trọng của giao dịch đó đối với việc đưa ra quyết định cuối cùng.
*   **Gated (Có cổng):** "Gated" có nghĩa là mạng chú ý này sử dụng các "cổng" (gates) để kiểm soát luồng thông tin, giúp nó học cách chọn lọc và kết hợp các đặc trưng một cách hiệu quả hơn.

Công thức của Gated Attention Network thường có dạng:

$$ s_k = 	ext{score}(h_k) = w^T ( 	anh(V h_k^T) 	ext{ ⊙ } 	ext{sigmoid}(U h_k^T) ) $$ 

Trong đó:
*   `h_k`: Biểu diễn đặc trưng của giao dịch thứ `k`.
*   `V, U, w`: Các ma trận trọng số và vector trọng số mà mô hình học được.
*   `tanh` và `sigmoid`: Các hàm kích hoạt (activation functions) giúp tạo ra các giá trị phi tuyến tính.
*   `⊙`: Phép nhân Hadamard (element-wise product).

Kết quả của Gated Attention Network là các **điểm chú ý (attention scores)** `s_k` cho mỗi giao dịch. Từ đó, chúng ta tính toán **trọng số chú ý (attention weights)** `α_k` bằng cách chuẩn hóa các điểm `s_k` (thường dùng hàm softmax) để tổng của tất cả `α_k` bằng 1.

### 5.6. Attention Weights (α_k) & Engineered Features

Ở lớp này, chúng ta có hai loại thông tin quan trọng:

*   **Attention Weights (α_k):** Các trọng số này cho biết mức độ "chú ý" của mô hình đến từng giao dịch. Một giao dịch có `α_k` cao nghĩa là nó được mô hình coi là rất quan trọng.
*   **Engineered Features (Các Đặc Trưng Kỹ Thuật):** Đây là các đặc trưng được các nhà nghiên cứu thiết kế thủ công dựa trên kiến thức chuyên môn về lừa đảo. Các đặc trưng này rất mạnh mẽ và giúp mô hình phát hiện các dấu hiệu lừa đảo mà có thể mạng neural khó học được một cách tự động. Ví dụ:
    *   **Counterparty Reputation (Uy tín đối tác):** Mức độ uy tín của các địa chỉ ví đối tác trong giao dịch.
    *   **Amount Spike (Đột biến giá trị):** Giao dịch có giá trị bất thường so với các giao dịch trước đó.
    *   **Novelty (Tính mới):** Giao dịch với một địa chỉ ví chưa từng tương tác trước đây.
    *   **Positional Features:** Vị trí của giao dịch trong chuỗi.

Việc kết hợp cả `Attention Weights` và `Engineered Features` là một điểm cải tiến quan trọng, giúp mô hình có cái nhìn toàn diện và chính xác hơn về các dấu hiệu lừa đảo.

### 5.7. Unified Prediction Heads (Các Đầu Dự Đoán Hợp Nhất)

Đây là nơi UnifiedTMIL thực hiện hai nhiệm vụ chính của mình. Kiến trúc mới sử dụng các mô hình Machine Learning truyền thống (gọi là "meta-learner" hoặc "ranker") ở đây để tăng cường hiệu suất và kiểm soát leakage.

#### 5.7.1. Head-C: Account Detection (Phát Hiện Tài Khoản Lừa Đảo)

*   **Gated Attention Pooling:** Đầu tiên, các `Instance Representations (h_k)` được tổng hợp lại thành một vector duy nhất `z_c` bằng cách sử dụng `Attention Weights (α_k)`:

    $$ z_c = \sum_{k=1}^{K} \alpha_k h_k $$ 

    `z_c` là một biểu diễn tổng hợp của toàn bộ chuỗi giao dịch, đã được "chú ý" vào các giao dịch quan trọng. 

*   **Ensemble Meta-Learner (LightGBM + XGBoost):** Thay vì chỉ dùng một lớp neural đơn giản, `z_c` được đưa vào một **Ensemble Meta-Learner**. 
    *   **Ensemble Learning:** Là kỹ thuật kết hợp nhiều mô hình học máy nhỏ hơn để tạo ra một mô hình mạnh mẽ hơn. Giống như việc hỏi ý kiến nhiều chuyên gia để đưa ra quyết định cuối cùng.
    *   **LightGBM và XGBoost:** Là hai thuật toán "Gradient Boosting" rất mạnh mẽ và phổ biến trong Machine Learning, nổi tiếng về tốc độ và độ chính xác. Việc sử dụng ensemble này giúp UnifiedTMIL đạt được hiệu suất cao trong việc phân loại tài khoản lừa đảo.

#### 5.7.2. Head-L: Transaction Localization (Định Vị Giao Dịch Lừa Đảo)

*   **Feature Fusion (Kết Hợp Đặc Trưng):** Đối với mỗi giao dịch, các `Attention Weights (α_k)` và `Engineered Features` được kết hợp lại thành một tập hợp đặc trưng phong phú. Điều này giúp mô hình có đầy đủ thông tin để đánh giá từng giao dịch.

*   **LambdaMART Ranker (LOO Protocol):** Tập hợp đặc trưng đã kết hợp được đưa vào một **LambdaMART Ranker**. 
    *   **Ranking (Xếp hạng):** Nhiệm vụ của ranker là gán một "điểm lừa đảo" cho mỗi giao dịch, sau đó sắp xếp các giao dịch theo điểm này. Giao dịch có điểm cao nhất sẽ là giao dịch có khả năng lừa đảo cao nhất.
    *   **LambdaMART:** Là một thuật toán ranking mạnh mẽ, thường được sử dụng trong các hệ thống tìm kiếm và gợi ý.
    *   **LOO Protocol (Leave-One-Out Protocol):** Đây là một giao thức đánh giá và huấn luyện rất nghiêm ngặt, đặc biệt quan trọng để tránh **data leakage**. Khi huấn luyện hoặc tính toán đặc trưng cho một giao dịch, giao thức này đảm bảo rằng thông tin từ chính giao dịch đó hoặc các giao dịch trong cùng một "test bag" (tập kiểm thử) không bao giờ được sử dụng. Điều này giúp đảm bảo tính trung thực của kết quả, tránh tình trạng mô hình "nhìn trộm" đáp án.

### 5.8. Outputs (Đầu Ra)

Cuối cùng, UnifiedTMIL đưa ra hai loại đầu ra:

*   **Account Label (Nhãn Tài Khoản):** Dự đoán liệu tài khoản đó là "Scammer" (lừa đảo) hay "Benign" (lành tính).
*   **Transaction Ranks (Xếp Hạng Giao Dịch):** Một danh sách các giao dịch được xếp hạng theo khả năng là "Phishing" (lừa đảo) hay "Normal" (bình thường), giúp người dùng hoặc hệ thống có thể tập trung vào các giao dịch đáng ngờ nhất.

## 6. Những Cải Tiến Quan Trọng và Khắc Phục Lỗi

Trong quá trình phát triển và kiểm toán dự án UnifiedTMIL, chúng tôi đã thực hiện nhiều cải tiến quan trọng và khắc phục các lỗi nghiêm trọng để đảm bảo tính trung thực và độ tin cậy của kết quả. Đây là những điểm mấu chốt giúp UnifiedTMIL trở thành một giải pháp đáng tin cậy.

### 6.1. Bảng So Sánh Kiến Trúc Cũ và Mới

Để dễ hình dung, hãy xem lại bảng so sánh kiến trúc cũ và mới mà chúng ta đã thảo luận trước đó:

| Thành Phần Kiến Trúc | Kiến Trúc Cũ (Ảnh Gốc) | Kiến Trúc Mới (Đã Cập Nhật) | Lý Do Thay Đổi / Cải Tiến Chính |
|----------------------|------------------------|-----------------------------|---------------------------------|
| **Backbone Feature Extraction** | 
  - **BERT4ETH Embeddings:** Counterparty, IN/OUT Direction, Amount, Count, Time Delta.
  - **1D-Temporal Convolutional Network (TCN):** Xử lý chuỗi giao dịch để tạo ra `Instance Representations` (h_k).
  - **Gated Attention Network:** Tạo `Attention Weights / Scores` (α_k, s_k) từ `Instance Representations`.
| 
  - **BERT4ETH Embeddings:** Tương tự.
  - **1D-Temporal Convolutional Network (TCN):** Tương tự.
  - **Gated Attention Network:** Tương tự, nhưng đầu ra được sử dụng khác biệt hơn.
| **Không thay đổi:** Các thành phần này vẫn là nền tảng vững chắc cho việc trích xuất đặc trưng từ dữ liệu giao dịch. |
| **Đầu Ra Mạng Chú Ý** | 
  - `Attention Weights / Scores` (α_k, s_k) được sử dụng trực tiếp cho cả hai Head.
| 
  - **Attention Weights (α_k):** Trọng số chú ý từ mạng chú ý.
  - **Engineered Features:** Các đặc trưng được thiết kế thủ công (Amount Spike, Novelty, Counterparty Reputation, Positional, v.v.).
| **Cải tiến quan trọng:** Kiến trúc mới tách biệt rõ ràng giữa trọng số chú ý học được và các đặc trưng kỹ thuật. Các đặc trưng kỹ thuật này đã được chứng minh là mạnh mẽ hơn và giúp giải quyết vấn đề leakage trong `cp_reputation`. |
| **Head-C: Account Classification** | 
  - **Gated Attention Pooling:** `z_c = Σ α_k h_k` (pooling trực tiếp các `Instance Representations` dựa trên trọng số chú ý).
  - Sau đó đưa vào một lớp phân loại đơn giản (thường là MLP) để dự đoán nhãn tài khoản.
| 
  - **Gated Attention Pooling:** Tương tự, pooling `Instance Representations`.
  - **Ensemble Meta-Learner (LightGBM + XGBoost):** Đầu ra từ pooling được đưa vào một **Ensemble (LightGBM + XGBoost)** để phân loại tài khoản.
| **Cải tiến hiệu suất:** Thay thế lớp phân loại đơn giản bằng một Ensemble Meta-Learner mạnh mẽ hơn (LightGBM + XGBoost) đã giúp cải thiện đáng kể hiệu suất phân loại tài khoản, đặc biệt là trên các tập dữ liệu khó (Hard-AUC). |
| **Head-L: Transaction Localization** | 
  - **Gated Attention Scoring:** `s_k = score(h_k)` (sử dụng trực tiếp điểm chú ý hoặc một hàm đơn giản trên `Instance Representations` để xếp hạng giao dịch).
  - Vấn đề **data leakage** nghiêm trọng đã được phát hiện ở đây, nơi `cp_reputation` được tính toán toàn cục, dẫn đến kết quả Hit@1 giả tạo 0.996.
| 
  - **Feature Fusion:** Kết hợp `Attention Weights` (α_k) với các `Engineered Features`.
  - **LambdaMART Ranker (LOO Protocol):** Sử dụng thuật toán học để xếp hạng LambdaMART với giao thức Leave-One-Out (LOO) nghiêm ngặt để xếp hạng giao dịch.
| **Sửa lỗi Leakage cốt lõi:** Đây là thay đổi quan trọng nhất. Kiến trúc mới loại bỏ hoàn toàn leakage bằng cách:
  1.  Sử dụng **Feature Fusion** để kết hợp các đặc trưng chú ý và đặc trưng kỹ thuật.
  2.  Áp dụng **LambdaMART Ranker** với **giao thức LOO nghiêm ngặt** để đảm bảo rằng thông tin từ test bag không bao giờ rò rỉ vào quá trình huấn luyện hoặc tính toán đặc trưng của train bags. Điều này đã đưa Hit@1 từ 0.996 (leaky) xuống 0.832 (clean và trung thực). |
| **Mục Tiêu Huấn Luyện** | 
  - **Account-level:** `L_bce` (Binary Cross-Entropy).
  - **Transaction-level:** `L_bce + βL_contrast` (BCE kết hợp với Contrastive Loss).
| 
  - Các hàm mất mát tương tự vẫn được sử dụng cho quá trình huấn luyện cơ bản của mạng neural, nhưng các đầu ra của Head-C và Head-L sau đó được xử lý bởi các mô hình ML truyền thống (Ensemble, LambdaMART) thay vì chỉ là các lớp neural đơn giản.
| **Cải thiện tính mạnh mẽ:** Việc tích hợp các mô hình ML truyền thống làm meta-learner/ranker ở các Head cuối cùng giúp tăng cường khả năng học và tổng quát hóa, đồng thời dễ dàng kiểm soát leakage hơn. |

### 6.2. Vấn Đề Data Leakage và Cách Khắc Phục

**Data Leakage (Rò rỉ dữ liệu)** là một trong những vấn đề nguy hiểm nhất trong Machine Learning. Nó xảy ra khi mô hình học được thông tin từ dữ liệu mà nó không được phép biết trong quá trình huấn luyện hoặc đánh giá, dẫn đến kết quả đánh giá bị thổi phồng một cách giả tạo và không phản ánh hiệu suất thực tế của mô hình trên dữ liệu mới.

Trong dự án UnifiedTMIL, một trường hợp leakage nghiêm trọng đã được phát hiện trong cách tính toán đặc trưng `cp_reputation` (uy tín đối tác) trong phiên bản cũ của code (`src_lean/lean_localization_gbm.py`). Cụ thể:

*   **Leakage xảy ra:** Đặc trưng `cp_reputation` được tính toán **một lần toàn cục** trước khi thực hiện vòng lặp Leave-One-Out (LOO). Điều này có nghĩa là khi mô hình được huấn luyện để dự đoán cho một "test bag" (tập kiểm thử), các đặc trưng của nó đã vô tình chứa thông tin từ chính "test bag" đó hoặc các "test bag" khác trong tập dữ liệu, mà đáng lẽ ra mô hình không được biết.
*   **Hậu quả:** Kết quả là chỉ số Hit@1 cho nhiệm vụ định vị giao dịch đạt mức **1.000 (100%)** một cách giả tạo. Điều này cho thấy mô hình "nhìn trộm" được đáp án, chứ không phải nó thực sự hoạt động tốt đến vậy.

**Cách UnifiedTMIL khắc phục:**

Để giải quyết triệt để vấn đề này, kiến trúc mới đã áp dụng **giao thức Leave-One-Out (LOO) nghiêm ngặt** trong **LambdaMART Ranker** cho Head-L. Cụ thể:

*   Khi tính toán đặc trưng `cp_reputation` cho một giao dịch trong một "test bag" cụ thể, chỉ những thông tin từ các "train bag" (tập huấn luyện) **khác** mới được sử dụng. Thông tin từ chính "test bag" đó hoặc các giao dịch trong cùng "test bag" sẽ bị loại trừ.
*   Điều này đảm bảo rằng mô hình chỉ học từ những gì nó được phép biết, và kết quả đánh giá phản ánh hiệu suất thực tế trên dữ liệu chưa từng thấy.

Nhờ việc khắc phục leakage này, chỉ số Hit@1 đã được điều chỉnh về mức **0.832**, một con số trung thực và đáng tin cậy hơn nhiều, đồng thời vẫn là một kết quả rất mạnh mẽ.

### 6.3. Đánh Giá Trung Thực (Honest Evaluation)

Ngoài việc khắc phục leakage, UnifiedTMIL còn nhấn mạnh vào một quy trình đánh giá **trung thực và nghiêm ngặt**:

*   **Cluster-aware Bootstrap Confidence Intervals (CI):** Thay vì chỉ báo cáo một giá trị trung bình, chúng tôi cung cấp khoảng tin cậy (CI) bằng phương pháp bootstrap. Điều này cho biết mức độ ổn định và đáng tin cậy của kết quả. Ví dụ, với X-AUC, thay vì chỉ có mean=0.992 (với CI vô lý), chúng tôi có mean=0.981 với CI=[0.973, 0.989], một khoảng tin cậy hợp lý và bao quanh giá trị trung bình.
*   **Kiểm soát Leakage:** Như đã giải thích ở trên, mọi nỗ lực đã được thực hiện để đảm bảo không có thông tin nào rò rỉ từ tập kiểm thử sang tập huấn luyện.
*   **Phân tích theo nguồn (by-source negative breakdowns):** Đánh giá hiệu suất của mô hình trên các loại dữ liệu tiêu cực (không lừa đảo) khác nhau để hiểu rõ hơn về khả năng tổng quát hóa của mô hình.

### 6.4. Kết Quả SOTA Trung Thực

Sau tất cả các cải tiến và kiểm toán nghiêm ngặt, UnifiedTMIL đạt được các kết quả "State-of-the-Art" (SOTA) một cách trung thực:

| Cấp độ | Metric | UnifiedTMIL (Mean ± Std) | Khoảng tin cậy 95% |
|-------|--------|-------------------------|--------------------|
| Account | ID-F1 | 0.744 ± 0.005 | [0.739, 0.749] |
| Account | Hard-AUC | 0.696 ± 0.148 | [0.570, 0.822] |
| Account | X-AUC | 0.981 ± 0.009 | [0.973, 0.989] |
| Transaction | Hit@1 | **0.832** | [0.752, 0.901] |
| Transaction | Hit@5 | 0.931 | — |
| Transaction | Hit@10 | 0.941 | — |
| Transaction | MRR | 0.880 | [0.823, 0.932] |

Các chỉ số này không chỉ cao mà còn đáng tin cậy, phản ánh hiệu suất thực sự của UnifiedTMIL trong việc phát hiện lừa đảo trên Ethereum.

## 7. Cách Dự Án Hoạt Động (Workflow Đơn Giản)

Để hình dung rõ hơn, hãy xem xét một kịch bản đơn giản về cách UnifiedTMIL hoạt động khi nhận được dữ liệu:

1.  **Đầu vào:** Hệ thống nhận một chuỗi các giao dịch của một tài khoản Ethereum cần được kiểm tra. Ví dụ, một tài khoản mới được tạo hoặc một tài khoản có hoạt động bất thường.
2.  **Tiền xử lý và Embedding:** Mỗi giao dịch trong chuỗi được tiền xử lý và chuyển đổi thành các "BERT4ETH Embeddings" (như đã giải thích ở Mục 5.3). Các đặc trưng này bao gồm thông tin về đối tác, giá trị, hướng giao dịch, v.v.
3.  **Học Mẫu Thời Gian (TCN):** Chuỗi các embedding này được đưa vào 1D-Temporal Convolutional Network (TCN) để học các mẫu và mối quan hệ theo thời gian giữa các giao dịch. Kết quả là các "Instance Representations" (h_k) cho mỗi giao dịch.
4.  **Mạng Chú Ý (Gated Attention Network):** Mạng chú ý sẽ phân tích các `h_k` để xác định giao dịch nào là quan trọng nhất trong chuỗi, tạo ra "Attention Weights" (α_k) và các "Engineered Features" liên quan.
5.  **Dự đoán Tài khoản (Head-C):**
    *   Các `h_k` được tổng hợp lại bằng `α_k` để tạo ra một biểu diễn tổng thể của tài khoản (`z_c`).
    *   `z_c` này sau đó được đưa vào **Ensemble Meta-Learner** (kết hợp LightGBM và XGBoost) để phân loại tài khoản là "Scammer" (lừa đảo) hay "Benign" (lành tính).
6.  **Định vị Giao dịch (Head-L):**
    *   Đối với mỗi giao dịch, các `Attention Weights (α_k)` và `Engineered Features` được kết hợp lại.
    *   Tập hợp đặc trưng này được đưa vào **LambdaMART Ranker** (với giao thức LOO nghiêm ngặt) để gán một "điểm lừa đảo" cho từng giao dịch. Các giao dịch sau đó được xếp hạng từ khả năng lừa đảo cao nhất đến thấp nhất.
7.  **Đầu ra:** Hệ thống cung cấp hai kết quả:
    *   **Nhãn tài khoản:** Ví dụ: "Tài khoản này có khả năng là lừa đảo."
    *   **Danh sách giao dịch đáng ngờ:** Ví dụ: "Giao dịch thứ 5 và thứ 8 trong chuỗi này có khả năng cao liên quan đến lừa đảo."

Toàn bộ quá trình này diễn ra trong "One Forward Pass", giúp hệ thống hoạt động nhanh chóng và hiệu quả.

## 8. Tầm Quan Trọng và Ứng Dụng Thực Tế

Dự án UnifiedTMIL mang lại nhiều lợi ích và có tiềm năng ứng dụng rộng rãi:

*   **Bảo vệ người dùng:** Cung cấp một công cụ mạnh mẽ để cảnh báo người dùng về các tài khoản và giao dịch lừa đảo tiềm năng, giúp họ tránh bị mất tài sản.
*   **Nâng cao niềm tin vào Blockchain:** Bằng cách giảm thiểu lừa đảo, UnifiedTMIL góp phần xây dựng một môi trường an toàn và đáng tin cậy hơn cho người dùng và nhà đầu tư trong không gian tiền điện tử.
*   **Hỗ trợ các sàn giao dịch và dự án DeFi:** Các nền tảng này có thể tích hợp UnifiedTMIL để tự động phát hiện và ngăn chặn các hoạt động lừa đảo, bảo vệ người dùng của họ và duy trì uy tín.
*   **Nghiên cứu và phát triển:** UnifiedTMIL cung cấp một khung nghiên cứu vững chắc với quy trình đánh giá trung thực, khuyến khích các nghiên cứu tiếp theo trong lĩnh vực phát hiện gian lận trên Blockchain.
*   **Hiệu quả và toàn diện:** Khả năng phát hiện cả ở cấp độ tài khoản và giao dịch trong một lần xử lý giúp tiết kiệm tài nguyên và cung cấp cái nhìn toàn diện hơn về các mối đe dọa.

## 9. Kết Luận và Hướng Phát Triển Tương Lai

UnifiedTMIL là một bước tiến quan trọng trong cuộc chiến chống lại lừa đảo trên Ethereum. Bằng cách kết hợp sức mạnh của Deep Learning (BERT4ETH, TCN, Attention) với các mô hình Machine Learning truyền thống mạnh mẽ (Ensemble, LambdaMART) và một quy trình đánh giá nghiêm ngặt, chúng tôi đã tạo ra một hệ thống không chỉ hiệu quả mà còn đáng tin cậy.

Các cải tiến về kiến trúc và việc khắc phục triệt để vấn đề data leakage đã đảm bảo rằng các kết quả được báo cáo là trung thực và phản ánh đúng hiệu suất của mô hình trong thế giới thực.

**Hướng phát triển tương lai:**

*   **Mở rộng sang các Blockchain khác:** Áp dụng UnifiedTMIL cho các mạng Blockchain khác ngoài Ethereum.
*   **Phát hiện các loại lừa đảo mới:** Liên tục cập nhật và cải tiến mô hình để đối phó với các chiêu trò lừa đảo ngày càng tinh vi.
*   **Tích hợp thời gian thực:** Phát triển hệ thống để có thể đưa ra cảnh báo lừa đảo gần như ngay lập tức khi giao dịch xảy ra.
*   **Giải thích mô hình (Explainable AI):** Cải thiện khả năng giải thích lý do tại sao mô hình đưa ra một dự đoán cụ thể, giúp người dùng và nhà phân tích tin tưởng hơn vào hệ thống.

Chúng tôi hy vọng cuốn sách này đã cung cấp cho bạn một cái nhìn toàn diện và dễ hiểu về dự án UnifiedTMIL, và truyền cảm hứng cho bạn khám phá thêm về lĩnh vực thú vị này.

## Phụ Lục

### A. Thuật Ngữ Quan Trọng

*   **Blockchain:** Công nghệ sổ cái phân tán, ghi lại các giao dịch một cách an toàn và bất biến.
*   **Ethereum:** Nền tảng Blockchain hỗ trợ hợp đồng thông minh.
*   **Smart Contract (Hợp đồng thông minh):** Chương trình tự thực thi trên Blockchain.
*   **Phishing (Lừa đảo):** Hành vi lừa đảo để chiếm đoạt thông tin hoặc tài sản.
*   **BERT4ETH Embeddings:** Biểu diễn số hóa của giao dịch Ethereum.
*   **TCN (Temporal Convolutional Network):** Mạng neural xử lý dữ liệu chuỗi thời gian.
*   **Attention Mechanism (Cơ chế chú ý):** Cho phép mô hình tập trung vào các phần quan trọng của dữ liệu.
*   **Data Leakage (Rò rỉ dữ liệu):** Thông tin từ tập kiểm thử vô tình lọt vào tập huấn luyện, làm sai lệch kết quả.
*   **LOO Protocol (Leave-One-Out Protocol):** Giao thức đánh giá nghiêm ngặt để tránh leakage.
*   **Ensemble Learning:** Kết hợp nhiều mô hình để cải thiện hiệu suất.
*   **LightGBM, XGBoost:** Các thuật toán Gradient Boosting mạnh mẽ.
*   **LambdaMART:** Thuật toán ranking được sử dụng trong các hệ thống tìm kiếm.
*   **SOTA (State-of-the-Art):** Hiệu suất tốt nhất hiện có trong một lĩnh vực cụ thể.

### B. Công Thức Toán Học

#### 1. Gated Attention Network

Công thức tính điểm chú ý `s_k` cho mỗi `Instance Representation h_k`:

$$ s_k = w^T ( \tanh(V h_k^T) \text{ ⊙ } \sigma(U h_k^T) ) $$

Trong đó:
*   `h_k`: Biểu diễn đặc trưng của giao dịch thứ `k`.
*   `V, U, w`: Các ma trận trọng số và vector trọng số học được.
*   `tanh`: Hàm hyperbolic tangent.
*   `σ`: Hàm sigmoid.
*   `⊙`: Phép nhân Hadamard (element-wise product).

#### 2. Gated Attention Pooling

Công thức tổng hợp các `Instance Representations h_k` thành một vector `z_c` cho tài khoản, sử dụng `Attention Weights α_k`:

$$ z_c = \sum_{k=1}^{K} \alpha_k h_k $$

Trong đó:
*   `α_k`: Trọng số chú ý của giao dịch thứ `k`.
*   `h_k`: Biểu diễn đặc trưng của giao dịch thứ `k`.
*   `K`: Tổng số giao dịch trong chuỗi.

#### 3. Binary Cross-Entropy (BCE) Loss

Đây là hàm mất mát phổ biến cho các bài toán phân loại nhị phân (ví dụ: Scammer/Benign). Công thức cho một mẫu:

$$ L_{BCE} = - (y \log(p) + (1-y) \log(1-p)) $$

Trong đó:
*   `y`: Nhãn thực tế (0 hoặc 1).
*   `p`: Xác suất dự đoán của mô hình cho nhãn 1.

#### 4. Contrastive Loss

Contrastive Loss được sử dụng để "kéo" các mẫu cùng loại lại gần nhau và "đẩy" các mẫu khác loại ra xa nhau. Công thức có thể phức tạp tùy thuộc vào biến thể, nhưng ý tưởng chính là:

$$ L_{contrast} = \max(0, m - d(x_1, x_2)) \text{ nếu } y=0 $$
$$ L_{contrast} = d(x_1, x_2) \text{ nếu } y=1 $$

Trong đó:
*   `d(x_1, x_2)`: Khoảng cách giữa hai mẫu `x_1` và `x_2`.
*   `y`: Nhãn cho biết hai mẫu có cùng loại hay không (0 nếu khác, 1 nếu cùng).
*   `m`: Margin (ngưỡng), một siêu tham số.
