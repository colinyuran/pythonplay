import urllib.request

with urlopen("http://vip.stock.finance.sina.com.cn/fund_center/data/jsonp.php/IO.XSRV2.CallbackList['eYW8jbaFYoDDwmVF']/NetValueReturn_Service.NetValueReturnOpen?page=4&num=40&sort=form_year&asc=0&ccode=&type2=0&type3=") as response:
    content = response.read()

    