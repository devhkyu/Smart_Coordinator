from icrawler.builtin import GoogleImageCrawler
# google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4, storage={'root_dir': '대학생룩'})
# 
# google_crawler.crawl(keyword='대학생룩', max_num=500, min_size=(200,200), max_size=None)
# 
# google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4, storage={'root_dir': '결혼식룩'})
# 
# google_crawler.crawl(keyword='결혼식룩', max_num=500, min_size=(200,200), max_size=None)
# 
# google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4, storage={'root_dir': '힙합룩'})
# 
# google_crawler.crawl(keyword='힙합패션', max_num=500, min_size=(200,200), max_size=None)
# 
# google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4, storage={'root_dir': '패션테러'})
# 
# google_crawler.crawl(keyword='패션테러', max_num=500, min_size=(200,200), max_size=None)

google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4, storage={'root_dir': '점프샷'})

google_crawler.crawl(keyword='점프샷', max_num=500, min_size=(200, 200), max_size=None)
